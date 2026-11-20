pub mod api;
pub mod build_info;
pub mod cancel {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::cancel::*;
    pub use ferx_model::cancel::{is_cancelled, CancelFlag};
}
#[cfg(feature = "survival")]
pub mod categorical {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::categorical::*;
    pub use ferx_model::categorical::{
        binary_diagnostics, binary_pearson_residual, count_diagnostics, count_pearson_residual,
        predict_binary, predict_count, simulate_binary, simulate_count,
    };
}
pub(crate) use ferx_model::cold_sort;
pub(crate) use ferx_model::covariate_stats;
pub mod diagnostics {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::diagnostics::*;
    pub use ferx_model::diagnostics::{
        first_error, CheckReport, Diagnostic, EngineError, Severity,
    };
}
pub(crate) use ferx_model::dosing;
pub mod edit {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::edit::*;
    pub use ferx_model::edit::{
        EliminationForm, ErrorForm, ErrorSpecText, EtaDecl, IivForm, InputForm, ModelEdit,
        ModelText, NewParameter, ParameterVariability, RandomEffectBlock, RandomEffectDecl,
        Relation, RelationTheta, SigmaDecl, StructuralEngine, StructuralSpec, ThetaDecl,
        TimeVaryingDecl, VariabilityText,
    };
}
pub(crate) use ferx_model::entry_refusals;
pub mod environment {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::environment::*;
    pub use ferx_model::environment::{detect, EnvironmentInfo};
}
pub mod estimation;
pub mod frem;
pub mod io;
pub(crate) use ferx_model::levels;
#[cfg(feature = "markov")]
pub mod markov {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::markov::*;
    pub use ferx_model::markov::{
        ctmm_data_term, ctmm_data_term_grad, ctmm_inhomogeneous_transition,
        ctmm_inhomogeneous_transition_with_opts, generator_rate_direction, matrix_exp,
        matrix_exp_frechet, MarkovError, StateObs,
    };
    pub mod endpoint {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::markov::endpoint::*;
        pub use ferx_model::markov::endpoint::{
            ctmm_subject_eta_grad, ctmm_subject_nll, validate_ctmm_states, validate_ctmm_times,
        };
    }
}
pub(crate) use ferx_model::model_checks;
pub mod model_selection;
pub mod nn {
    pub use ferx_model::nn::Activation;
    #[allow(unused_imports)]
    pub(crate) use ferx_model::nn::*;
    #[cfg(feature = "nn")]
    pub use ferx_model::nn::{CovariateMapper, CovariateNn, MlpMapper, NamedMlpMapper, NnError};
    pub mod export {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::nn::export::*;
        pub use ferx_model::nn::export::{
            export_dcm_as_analytic, export_dcm_from_fit, DcmExportOptions, DcmWeightForm,
        };
    }

    // Lives here, not in ferx-model: it embeds a repo-level fixture, which a published
    // ferx-model package does not contain (#3041), and only ferx-core tests call it.
    /// `tests/fixtures/two_cpt_dcm_regularized.ferx` with its `init` line removed:
    /// every output-layer bias at 0, so each output starts at `softplus(0)`. This is
    /// the start #1277's line-search abort was measured from, and the one #1561 found
    /// reporting convergence far above the covariate-free point.
    ///
    /// The penalty-kernel and gradient-parity tests evaluate at this start too, for a
    /// different reason: its output-layer weights are Glorot draws. A declared `init`
    /// zeroes that block, so the network is flat in its inputs, the curvature penalty
    /// and its gradient read ≈ 0, and a parity check would compare zeros.
    ///
    /// Panics unless exactly one line was stripped, so a reworded `init` line cannot
    /// quietly turn this back into the declared start.
    #[cfg(test)]
    pub(crate) fn dcm_fixture_without_init() -> String {
        const MODEL: &str = include_str!("../tests/fixtures/two_cpt_dcm_regularized.ferx");
        let bare: String = MODEL
            .lines()
            .filter(|l| !l.trim_start().starts_with("init "))
            .map(|l| format!("{l}\n"))
            .collect();
        assert_eq!(
            bare.lines().count() + 1,
            MODEL.lines().count(),
            "exactly one `init` line must be stripped from the DCM fixture"
        );
        bare
    }
}
pub(crate) use ferx_model::occasions;
pub mod ode {
    #[cfg(feature = "survival")]
    pub use ferx_model::ode::ThresholdOutcome;
    #[allow(unused_imports)]
    pub(crate) use ferx_model::ode::*;
    pub use ferx_model::ode::{
        ode_dense_solve_states, ode_predictions, ode_predictions_ekf_with_diffusion,
        ode_predictions_event_driven, ode_predictions_event_driven_with_states,
        ode_predictions_with_solver_stats, ode_predictions_with_states, solve_ode, solve_ode_g,
        solve_ode_g_dense, solve_ode_g_with_stats, solve_ode_until_threshold, solve_ode_with_stats,
        OdeMethod, OdeOutputFn, OdeReadout, OdeRhsFn, OdeSolverOptions, OdeSolverStats, OdeSpec,
        PerCmtReadout, SolPoint, SolPointG, ThresholdCrossing,
    };
    pub mod predictions {
        #[cfg(feature = "survival")]
        pub use ferx_model::ode::predictions::ThresholdOutcome;
        #[allow(unused_imports)]
        pub(crate) use ferx_model::ode::predictions::*;
        pub use ferx_model::ode::predictions::{
            ode_dense_solve_states, ode_predictions, ode_predictions_ekf_with_diffusion,
            ode_predictions_event_driven, ode_predictions_event_driven_with_states,
            ode_predictions_with_solver_stats, ode_predictions_with_states, OdeOutputFn,
            OdeReadout, OdeSpec, PerCmtReadout,
        };
    }
    pub mod solver {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::ode::solver::*;
        pub use ferx_model::ode::solver::{
            solve_ode, solve_ode_g, solve_ode_g_dense, solve_ode_g_with_stats,
            solve_ode_until_threshold, solve_ode_with_stats, OdeMethod, OdeRhsFn, OdeSolverOptions,
            OdeSolverStats, SolPoint, SolPointG, ThresholdCrossing,
        };
    }
    pub mod stiffness {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::ode::stiffness::*;
        pub use ferx_model::ode::stiffness::{STIFF_RE_LAMBDA_THRESHOLD, STIFF_TAU_FAST};
    }
}
pub(crate) use ferx_model::par;
pub mod parser {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::parser::*;
    pub mod covariate_model {
        pub use ferx_model::parser::covariate_model::CovariateStatBindings;
        #[allow(unused_imports)]
        pub(crate) use ferx_model::parser::covariate_model::*;
    }
    pub mod model_parser {
        #[cfg(feature = "markov")]
        pub use ferx_model::parser::model_parser::CtmmGeneratorProgram;
        #[cfg(feature = "survival")]
        pub use ferx_model::parser::model_parser::EndpointExprProgram;
        #[allow(unused_imports)]
        pub(crate) use ferx_model::parser::model_parser::*;
        pub use ferx_model::parser::model_parser::{
            apply_fit_option, apply_method_option, known_block_names, parse_method_token,
            DataBindings, IndivParamPartials, IndivParamProgram, LevelBinding, LevelBindings,
            LevelBlockDecl, LevelContrast, OdeOutputProgram, OdeRhsProgram, ParseBindings,
            RuvMagDerivProgram, ScaleDerivProgram, ThetaBlocks, NN_WEIGHT_DIRECT_REFERENCE_MARKER,
            ODE_INIT_REJECTED_BUILTINS, ODE_INIT_SCOPE_BUILTINS,
        };
        // The entry points that can reach `[priors] from_fit`: each installs the `.fitrx`
        // reader, which lives here, before handing over to the parser (#3041).
        /// Parse a model file (.ferx) and return a CompiledModel.
        ///
        /// Goes through [`parse_full_model_file`] rather than [`parse_model_string`] so
        /// the relative-path rewrites that need the model file's directory —
        /// `[data] path` and `[priors] from_fit` — apply here too. A caller that only
        /// wants the `CompiledModel` still gets a `from_fit` resolved against the model,
        /// not against the process's working directory.
        pub fn parse_model_file(
            path: &std::path::Path,
        ) -> Result<ferx_model::types::CompiledModel, String> {
            crate::io::fitrx::install_fit_estimates_reader();
            ferx_model::parser::model_parser::parse_model_file(path)
        }
        /// Parse a full model file including simulation spec, initial values, and fit options.
        pub fn parse_full_model_file(
            path: &std::path::Path,
        ) -> Result<ferx_model::types::ParsedModel, String> {
            crate::io::fitrx::install_fit_estimates_reader();
            ferx_model::parser::model_parser::parse_full_model_file(path)
        }
        /// Read `model.prior_from_fit` and turn it into ordinary
        /// [`crate::types::ParameterPrior`]s on `model.priors` (#254 phase 2).
        ///
        /// Every parse entry point calls this, so a model that came from
        /// [`parse_model_file`] / [`parse_model_string`] is already expanded and calling
        /// it again is a no-op. It is public for the other producer of a
        /// `CompiledModel` — a caller that builds one field by field (ferx-r) and sets
        /// `prior_from_fit` itself has nothing to read the fit file for it, and
        /// `PriorSet::build` refuses an unexpanded model rather than fitting it
        /// unpenalized.
        ///
        /// A relative path resolves against the process's working directory; resolve it
        /// yourself first if it should be relative to something else.
        pub fn expand_prior_from_fit(
            model: &mut ferx_model::types::CompiledModel,
        ) -> Result<(), String> {
            crate::io::fitrx::install_fit_estimates_reader();
            ferx_model::parser::model_parser::expand_prior_from_fit(model)
        }
        /// Parse a model string and return a CompiledModel (backward compatible).
        pub fn parse_model_string(
            content: &str,
        ) -> Result<ferx_model::types::CompiledModel, String> {
            crate::io::fitrx::install_fit_estimates_reader();
            ferx_model::parser::model_parser::parse_model_string(content)
        }
        /// Parse a full model string including all optional blocks.
        pub fn parse_full_model(content: &str) -> Result<ferx_model::types::ParsedModel, String> {
            crate::io::fitrx::install_fit_estimates_reader();
            ferx_model::parser::model_parser::parse_full_model(content)
        }
        /// [`parse_full_model`] with the observed levels of every `theta NAME ~
        /// NAME[COL, ...]` block supplied (#1064).
        ///
        /// A level block's level count is a property of the *data*, which the first
        /// parse has not seen — so `levels::bind` reads the population, discovers
        /// the observed combinations, and calls this. Re-parsing (milliseconds) is what
        /// lets the compiled closures be built once, with the real θ count, instead of
        /// being rebuilt in place after the fact.
        pub fn parse_full_model_bound(
            content: &str, level_bindings: &ferx_model::parser::model_parser::LevelBindings,
        ) -> Result<ferx_model::types::ParsedModel, String> {
            crate::io::fitrx::install_fit_estimates_reader();
            ferx_model::parser::model_parser::parse_full_model_bound(content, level_bindings)
        }
        /// [`parse_full_model`] with every data-derived binding supplied.
        pub fn parse_full_model_with(
            content: &str, bindings: &ferx_model::parser::model_parser::ParseBindings,
        ) -> Result<ferx_model::types::ParsedModel, String> {
            crate::io::fitrx::install_fit_estimates_reader();
            ferx_model::parser::model_parser::parse_full_model_with(content, bindings)
        }
    }
}
pub mod pk {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::pk::*;
    pub use ferx_model::pk::{
        add_analytical_init, analytical_state_at_times, apply_analytic_readout, apply_scaling,
        compute_event_pk_params, compute_event_pk_params_into, compute_predictions,
        compute_predictions_ode, compute_predictions_with_states, compute_predictions_with_tv,
        compute_predictions_with_tv_into, compute_predictions_with_tv_into_with_schedule,
        one_cpt_ig_f, one_cpt_infusion, one_cpt_infusion_ss, one_cpt_iv_bolus, one_cpt_iv_bolus_ss,
        one_cpt_oral, one_cpt_oral_f, one_cpt_oral_f_ss, one_cpt_oral_ss, one_cpt_predict,
        one_cpt_transit_f, predict_all_states, predict_concentration, predict_iov,
        three_cpt_infusion, three_cpt_infusion_ss, three_cpt_iv_bolus, three_cpt_iv_bolus_ss,
        three_cpt_oral, three_cpt_oral_f, three_cpt_oral_f_ss, three_cpt_oral_ss, two_cpt_ig_f,
        two_cpt_infusion, two_cpt_infusion_ss, two_cpt_iv_bolus, two_cpt_iv_bolus_ss, two_cpt_oral,
        two_cpt_oral_f, two_cpt_oral_f_ss, two_cpt_oral_ss, two_cpt_predict, two_cpt_transit_f,
        validate_per_cmt_scaling, EventPkParams,
    };
    pub mod absorption {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::pk::absorption::*;
        pub use ferx_model::pk::absorption::{
            validate_first_order, validate_igd, validate_transit, validate_weibull,
            validate_zero_order, InputRateForcing, InputRateKind, PreparedInputRate,
        };
    }
    pub mod analytical_absorption {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::pk::analytical_absorption::*;
        pub use ferx_model::pk::analytical_absorption::{
            convolve_1cpt, convolve_2cpt, IgAbsorption, TiltedAbsorption, TransitAbsorption,
        };
    }
    pub mod event_driven {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::pk::event_driven::*;
        pub use ferx_model::pk::event_driven::{
            event_driven_predictions, event_driven_predictions_with_schedule, profile_report,
            supports_event_driven, Event, EventKind, EventSchedule,
        };
    }
    pub mod ode_template {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::pk::ode_template::*;
        pub use ferx_model::pk::ode_template::{
            generate, generate_variant, EliminationForm, GeneratedDisposition, InputForm,
        };
    }
    pub mod one_compartment {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::pk::one_compartment::*;
        pub use ferx_model::pk::one_compartment::{
            one_cpt_ig_f, one_cpt_infusion, one_cpt_infusion_ss, one_cpt_iv_bolus,
            one_cpt_iv_bolus_ss, one_cpt_oral, one_cpt_oral_f, one_cpt_oral_f_ss, one_cpt_oral_ss,
            one_cpt_predict, one_cpt_transit_f,
        };
    }
    pub mod three_compartment {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::pk::three_compartment::*;
        pub use ferx_model::pk::three_compartment::{
            three_cpt_infusion, three_cpt_infusion_ss, three_cpt_iv_bolus, three_cpt_iv_bolus_ss,
            three_cpt_oral, three_cpt_oral_f, three_cpt_oral_f_ss, three_cpt_oral_ss,
        };
    }
    pub mod two_compartment {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::pk::two_compartment::*;
        pub use ferx_model::pk::two_compartment::{
            two_cpt_ig_f, two_cpt_infusion, two_cpt_infusion_ss, two_cpt_iv_bolus,
            two_cpt_iv_bolus_ss, two_cpt_oral, two_cpt_oral_f, two_cpt_oral_f_ss, two_cpt_oral_ss,
            two_cpt_predict, two_cpt_transit_f,
        };
    }
}
pub(crate) mod pool;
pub mod propensity_match;
pub(crate) use ferx_model::rng;
pub(crate) use ferx_model::scratch;
pub mod sens {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::sens::*;
    pub mod dual2 {
        pub use ferx_model::sens::dual2::Dual2;
        #[allow(unused_imports)]
        pub(crate) use ferx_model::sens::dual2::*;
    }
    pub mod num {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::sens::num::*;
        pub use ferx_model::sens::num::{BranchCmp, HostFn, JetFn, NonRecording, PkNum};
    }
    pub mod ode_provider {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::sens::ode_provider::*;
    }
    pub mod one_cpt {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::sens::one_cpt::*;
        pub use ferx_model::sens::one_cpt::{
            one_cpt_conc_g, one_cpt_ig_amt_g, one_cpt_ig_conc_g, one_cpt_ig_g, one_cpt_infusion_g,
            one_cpt_infusion_ss_g, one_cpt_iv_bolus_amt_g, one_cpt_iv_bolus_g,
            one_cpt_iv_bolus_ss_g, one_cpt_oral_amt_g, one_cpt_oral_g, one_cpt_oral_ss_g,
            one_cpt_transit_amt_g, one_cpt_transit_conc_g, one_cpt_transit_g, one_cpt_zero_order_g,
        };
    }
    pub mod propagate {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::sens::propagate::*;
        pub use ferx_model::sens::propagate::{
            event_driven_sens_g, event_driven_sens_with_doses_g, propagate_one_cpt_g,
            propagate_one_cpt_oral_g, propagate_three_cpt_core_g, propagate_three_cpt_g,
            propagate_three_cpt_oral_core_g, propagate_three_cpt_oral_g, propagate_two_cpt_core_g,
            propagate_two_cpt_g, propagate_two_cpt_oral_core_g, propagate_two_cpt_oral_g,
            three_cpt_eigen_g, two_cpt_eigen_g, EigenCacheG, OneCptPk, PkDual, ThreeCptEigen,
            ThreeCptModeG, TwoCptEigen,
        };
    }
    pub mod provider {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::sens::provider::*;
        pub use ferx_model::sens::provider::{
            analytic_outer_gradient_available, subject_eta_grad, subject_eta_jacobian, ObsGrad,
        };
    }
    pub mod three_cpt {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::sens::three_cpt::*;
        pub use ferx_model::sens::three_cpt::{
            three_cpt_infusion_g, three_cpt_infusion_ss_g, three_cpt_iv_bolus_amt_g,
            three_cpt_iv_bolus_g, three_cpt_iv_bolus_ss_g, three_cpt_oral_amt_g, three_cpt_oral_g,
            three_cpt_oral_ss_g,
        };
    }
    pub mod two_cpt {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::sens::two_cpt::*;
        pub use ferx_model::sens::two_cpt::{
            two_cpt_ig_amt_g, two_cpt_ig_conc_g, two_cpt_ig_g, two_cpt_infusion_g,
            two_cpt_infusion_ss_g, two_cpt_iv_bolus_amt_g, two_cpt_iv_bolus_g,
            two_cpt_iv_bolus_ss_g, two_cpt_oral_amt_g, two_cpt_oral_g, two_cpt_oral_ss_g,
            two_cpt_transit_amt_g, two_cpt_transit_conc_g, two_cpt_transit_g, two_cpt_zero_order_g,
        };
    }
}
pub mod sim;
pub(crate) use ferx_model::sim_result;
pub mod stats;
pub mod suggest_start;
#[cfg(feature = "survival")]
pub mod survival {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::survival::*;
    pub use ferx_model::survival::{
        cum_hazard, hazard_and_cum_hazard, mean_survival, median_survival,
        rtte_forward_nll_from_curves, rtte_reset_nll_from_curves, sample_conditional_event_time,
        sample_event_time, MonoTol,
    };
    pub mod parametric {
        #[allow(unused_imports)]
        pub(crate) use ferx_model::survival::parametric::*;
        pub use ferx_model::survival::parametric::{
            cum_hazard, hazard_and_cum_hazard, mean_survival, median_survival,
            sample_conditional_event_time, sample_event_time,
        };
    }
}
pub mod types {
    #[cfg(feature = "nn")]
    pub use ferx_model::types::NeuralNetworkInfo;
    #[allow(unused_imports)]
    pub(crate) use ferx_model::types::*;
    pub use ferx_model::types::{
        classify_warning, framework_keys, method_specific_keys, omega_se_at,
        AbsorptionOdeEquivalent, AggFunction, AnalyticReadout, AnalyticalInit, BayesResult,
        BicInputs, BloqMethod, CmtToggle, CompiledModel, CondDist, CovarianceFallback,
        CovarianceMethod, CovarianceStatus, CovariateDecl, CovariateForm, CovariateKind,
        CovariateLevels, CovariateModelSpec, CovariateMuRef, CovariateOp, CovariateRelation,
        CovariateRelationEstimate, CovariateRow, CovariateStat, CovariateSummary, CovariateTable,
        CovariateTheta, CovariateThetaEstimate, DerivedContext, DerivedEvalFn, DerivedExprSpec,
        DerivedFilterFn, DerivedKind, DiscreteObsDiagnostic, DoseAttr, DoseAttrConsumption,
        DoseAttrMap, DoseEvent, EndpointError, EndpointPredictionResult, ErrorModel, ErrorSelectFn,
        ErrorSelector, ErrorSpec, EstimationMethod, EtaParamInfo, EtaParamType, ExclusionSummary,
        FitOptions, FitResult, FitStatus, FremConfig, GradientMethod, HeldAtBound, HeldReason,
        HessianAnchor, ImpmapTrace, ImpmapTraceRow, ImportanceSamplingResult, IndivParamPartials,
        InfusionDef, InnerOptimizer, IntegralStep, IntegralWindow, IovOccasionRule, KappaTreatment,
        KappaWeight, KappaWeightFn, LevelGroupEbe, LevelKappaEbes, MixingExpr,
        MixtureClassOverride, MixtureMuRef, MixtureOverrideParam, MixtureParams, MixtureSpec,
        ModelParameters, MstepOptimizer, MuRef, MuTransform, ObsRecord, OmegaMatrix, Optimizer,
        ParameterKind, ParameterPrior, ParameterScaling, ParsedModel, PkModel, PkParamFn, PkParams,
        PlanId, Population, PopulationFingerprint, PosteriorSummary, Prediction, PriorSpread,
        PriorSummary, RateMode, ResidualCorrelation, ResidualSerial, ResidualSerialKind, RuvMagFn,
        RuvMagnitude, SaemKernel, SaemMstepSolver, SaemObjective, SaemOfv, ScaleAdaptation,
        ScaleFn, ScalingSpec, SigmaType, SigmaVector, SimOutcome, SimulationSpec, SirScale,
        StartRecord, StatusReason, Subject, SubjectResult, ThetaBlocks, ThetaTransform, TvFn,
        ViEtaGrad, ViFamily, ViFinalOfv, ViKl, ViOmegaUpdate, ViResult, ViSigmaUpdate, WarningCode,
        WarningEntry, WarningSeverity, BOBYQA_MAX_DIM, COV_HESSIAN_MAX_DIM, MAX_PK_PARAMS,
        PK_IDX_CL, PK_IDX_CV2, PK_IDX_F, PK_IDX_KA, PK_IDX_LAGTIME, PK_IDX_MAT, PK_IDX_MTT,
        PK_IDX_N, PK_IDX_Q, PK_IDX_Q3, PK_IDX_V, PK_IDX_V2, PK_IDX_V3,
    };
    #[cfg(feature = "survival")]
    pub use ferx_model::types::{
        CountFamily, CountMeanScale, CountParamFn, EndpointLikelihood, EventType, HazardFamily,
        HazardParamFn, HazardSpec, LinearPredictorFn, LinkFn, RtteClock, TteObsDiagnostic,
        TteRecurrence,
    };
    #[cfg(feature = "markov")]
    pub use ferx_model::types::{GeneratorFn, TransitionProbFn};
}

// Deprecated in favour of `bind_from_fit` (#1619); kept for callers pinned to it.
#[allow(deprecated)]
pub use api::bind_theta_levels_from_fit;
pub use api::{
    bind_from_fit, bind_theta_levels, check_model_data, check_model_data_warnings,
    check_model_options, configure_global_thread_pool, fit, fit_from_files, install_on_engine_pool,
    layout_from_fit, predict, predict_diag, prepare_run, prepare_run_with_inits,
    read_population_with, read_population_with_source, resolve_data_path, run_model_simulate,
    run_model_simulate_with_overrides, run_model_with_data, run_model_with_data_inits,
    run_model_with_overrides, simulate, simulate_adaptive, simulate_adaptive_from_spec,
    simulate_adaptive_from_spec_with_inputs, simulate_adaptive_with_inputs, simulate_with_options,
    simulate_with_options_diag, simulate_with_seed, simulate_with_uncertainty,
    simulate_with_uncertainty_diag, theta_level_map, theta_level_values, validate_model_file,
    validate_model_file_with, AdaptiveInputs, AdaptiveSimulateOptions, AdaptiveSimulationResult,
    DrawnParameters, PoolPlan, PredictionOutput, PredictionResult, PreparedRun, ReaderSettings,
    RunOverrides, SimulateOptions, SimulateUncertaintyOptions, SimulatedParameters,
    SimulatedSubjectParameters, SimulationOutput, SimulationResult, SuppliedEtas, UncertaintySpec,
    UnmatchedSubjects, FIT_RAYON_STACK_SIZE,
};
pub use cancel::CancelFlag;
pub use diagnostics::{CheckReport, Diagnostic, EngineError, Severity};
pub use environment::EnvironmentInfo;
pub use estimation::fit_inputs::ScoringSettings;
pub use estimation::run_covariance::run_covariance;
pub use estimation::run_sir::run_sir;
pub use estimation::uncertainty_samples::UncertaintyMethod;
pub use frem::{prepare_frem, FremDataInfo, FremFitInit, FremPrepareResult};
pub use io::data_source::DataSource;
pub use io::datareader::{read_nonmem_csv, read_nonmem_csv_with_covariates};
pub use io::model_source::ModelSource;
pub use model_selection::{
    bic, check_strictness, estimate_near_boundary, max_abs_correlation, natural_scale_covariance,
    stalled_at_init, BicType, Strictness, StrictnessVerdict,
};
pub use parser::model_parser::{
    apply_method_option, known_block_names, parse_full_model_file, parse_method_token,
    parse_model_file, parse_model_string, ODE_INIT_REJECTED_BUILTINS, ODE_INIT_SCOPE_BUILTINS,
};
pub use propensity_match::MatchMethod;
// Adaptive (feedback) dosing vocabulary (#391). Re-exported at the crate root so
// the public `simulate_adaptive` API — its controller, monitors, and the fields
// of `AdaptiveSimulationResult` (`ledger` / `decisions`) — is usable without
// reaching into the `sim::adaptive` module path.
pub use sim::adaptive::{
    AdaptiveAction, AdaptiveDosingSpec, AdaptiveRoute, AdaptiveRule, AdaptiveSubjectMetrics,
    Comparison, ControllerCtx, DecisionLogEntry, DecisionOutcome, DoseAction, DoseLedgerEntry,
    DoseStep, IntervalStep, MonitorSpec, ObserveMode, ObservedSignal,
};
pub use suggest_start::{inits_from_nca, NcaInit, SuggestedStart};
pub use types::*;

#[cfg(feature = "survival")]
pub use api::{
    predict_categorical, predict_survival, predict_survival_at_etas_diag, predict_survival_diag,
    SurvivalPredictionOutput, SurvivalPredictionResult,
};
// Kaplan–Meier mean covariate VPC of a single-event TTE endpoint (#1886).
#[cfg(feature = "survival")]
pub use api::{vpc_kmmc, KmmcVpc, KmmcVpcStratum, VpcKmmcOptions};
// VPC of a repeated-events endpoint: mean cumulative function and Kaplan–Meier by
// event order (#1886).
#[cfg(feature = "survival")]
pub use api::{vpc_rtte, RtteMcfBand, RtteOrderBand, RtteVpc, RtteVpcStratum, VpcRtteOptions};
// Kaplan–Meier VPC of a single-event TTE endpoint (#1886).
#[cfg(feature = "survival")]
pub use api::{
    vpc_tte, TteVpc, TteVpcStratum, VpcCensoring, VpcGrid, VpcStratify, VpcTteOptions,
    VPC_TTE_MAX_STRATA,
};

/// Positive proof that the `debug_assert!` guards in this crate are LIVE
/// (#344, #1248).
///
/// `release`, `ci-test` and `ci-fast` all leave `debug-assertions` off, so all ~180
/// `debug_assert!`s compile to nothing there. `[profile.ci-cov]` turns them on, and
/// the two per-PR coverage jobs build under it — which both runs the guards and
/// stops their condition lines being regions that can never execute, the
/// permanently-missed patch lines of #1248.
///
/// That is an invariant held by *absence*, which is the fragile kind. Nothing in a
/// test result distinguishes a build with the guards live from one without: a
/// `[profile.ci-cov] debug-assertions = false` in `Cargo.toml`, a
/// `CARGO_PROFILE_CI_COV_DEBUG_ASSERTIONS=false` in the workflow environment, or a
/// `--profile` that quietly reverted to `ci-fast` would neuter the whole thing while
/// all 4366 tests and every preflight-contract test stayed green. The gate would
/// then be exactly the no-op #344 was filed to replace.
///
/// So each caller arms this canary explicitly — `FERX_REQUIRE_DEBUG_ASSERTIONS=1`,
/// on the coverage steps in `ci.yml` and in the argument vector of
/// `tools/preflight.sh`'s `debug-assertions` group (visible in `--list`, and
/// asserted by `tests/preflight_owns_the_fast_gates.rs`) — and the canary fails the
/// run when the guards turn out to be dead.
#[cfg(test)]
#[path = "lib_tests.rs"]
mod debug_assertion_canary;

/// Model texts shared by test modules (written once instead of per test).
#[cfg(test)]
pub(crate) use ferx_model::test_models;

#[cfg(test)]
mod model_tests;
