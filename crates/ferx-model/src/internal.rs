//! Sealed access for ferx-core to the crate-private members of ferx-model's public types (#3041).
//!
//! ferx-core used these as `pub(crate)` items before the crate split. They stay crate-private
//! here; each trait below re-offers exactly the ones ferx-core calls, by the same name and
//! signature, so its call sites read as before once it has `use ferx_model::internal::*`.
//! ferx-core never re-exports this module, so none of it is ferx-core public API, and the
//! traits are sealed: nothing outside this crate can implement them.
#![allow(clippy::too_many_arguments, clippy::wrong_self_convention, missing_docs)]
#![allow(unused_imports)]
use crate::diagnostics::Diagnostic;
use crate::diagnostics::EngineError;
use crate::estimation::population_check::OutOfDomain;
use crate::estimation::population_check::ScoringSettings;
use crate::estimation::sir_settings::SirSettings;
use crate::io::datareader::SelectionFilter;
use crate::ode::predictions::OdeSpec;
use crate::ode::solver::OdeSolverStats;
use crate::parser::model_parser::Bytecode;
#[cfg(feature = "markov")]
use crate::parser::model_parser::CtmmGeneratorProgram;
#[cfg(feature = "survival")]
use crate::parser::model_parser::EndpointExprProgram;
use crate::parser::model_parser::EtaCoupling;
use crate::parser::model_parser::Expression;
use crate::parser::model_parser::IndivParamPartials;
use crate::parser::model_parser::IndivParamProgram;
use crate::parser::model_parser::LevelBlockDecl;
use crate::parser::model_parser::LevelContrast;
use crate::parser::model_parser::MuFoldSpec;
use crate::parser::model_parser::OdeOutputProgram;
use crate::parser::model_parser::OdeRhsProgram;
use crate::parser::model_parser::RhsBindingPlan;
use crate::parser::model_parser::RhsHoist;
use crate::parser::model_parser::RuvMagDerivProgram;
use crate::parser::model_parser::ScaleDerivProgram;
use crate::parser::model_parser::Statement;
use crate::parser::model_parser::ThetaBlocks;
use crate::parser::model_parser::VectorThetaDecl;
use crate::pk::topology::PkTopology;
use crate::sens::dual2::Dual2;
use crate::sim::adaptive::AbsorbedIncrease;
use crate::sim::adaptive::AdaptiveAction;
use crate::sim::adaptive::AdaptiveDosingSpec;
use crate::sim::adaptive::AdaptiveRule;
use crate::sim::adaptive::DoseGrid;
use crate::sim::adaptive::IntervalStep;
use crate::sim::adaptive::SubjectDosePlan;
use crate::types::population_fingerprint::Difference;
use crate::types::population_fingerprint::PopulationFingerprint;
use crate::types::AbsorptionOdeEquivalent;
use crate::types::AnalyticReadout;
use crate::types::BloqMethod;
use crate::types::CompiledModel;
use crate::types::CovariateMuRef;
use crate::types::DoseAttrMap;
use crate::types::DoseEvent;
use crate::types::ErrorSpec;
use crate::types::EstimationMethod;
use crate::types::FitOptions;
use crate::types::GradientMethod;
use crate::types::OmegaMatrix;
use crate::types::Optimizer;
use crate::types::PkModel;
use crate::types::PlanId;
use crate::types::RuvMagnitude;
use crate::types::Subject;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use std::collections::{BTreeMap, HashMap, HashSet};

mod sealed {
    pub trait Sealed {}
}

pub trait EngineErrorInternal: sealed::Sealed + Sized {
    fn from_diagnostic(d: Diagnostic) -> Self;
    fn from_diagnostic_keeping_text(d: Diagnostic, message: String) -> Self;
    fn with_suggestion_in_display(d: Diagnostic) -> Self;
    fn uncoded(message: impl Into<String>) -> Self;
    fn in_context(self, context: impl Into<String>) -> Self;
}
impl EngineErrorInternal for EngineError {
    fn from_diagnostic(d: Diagnostic) -> Self {
        EngineError::from_diagnostic(d)
    }
    fn from_diagnostic_keeping_text(d: Diagnostic, message: String) -> Self {
        EngineError::from_diagnostic_keeping_text(d, message)
    }
    fn with_suggestion_in_display(d: Diagnostic) -> Self {
        EngineError::with_suggestion_in_display(d)
    }
    fn uncoded(message: impl Into<String>) -> Self {
        EngineError::uncoded(message)
    }
    fn in_context(self, context: impl Into<String>) -> Self {
        EngineError::in_context(self, context)
    }
}
impl sealed::Sealed for EngineError {}

pub trait DoseEventInternal: sealed::Sealed + Sized {
    const DURATION_FLOOR: f64;
    const RATE_FLOOR: f64;
    fn resolve_rate(&self, attr_map: &DoseAttrMap, params: &[f64]) -> DoseEvent;
    fn cmt_raw(&self) -> usize;
    fn bioavailable_infusion(&self, f_bio: f64) -> (f64, f64);
    fn with_bioavailable_infusion(&self, f_bio: f64) -> DoseEvent;
    fn ss_cycle_doses(&self) -> Vec<DoseEvent>;
}
impl DoseEventInternal for DoseEvent {
    const DURATION_FLOOR: f64 = DoseEvent::DURATION_FLOOR;
    const RATE_FLOOR: f64 = DoseEvent::RATE_FLOOR;
    fn resolve_rate(&self, attr_map: &DoseAttrMap, params: &[f64]) -> DoseEvent {
        DoseEvent::resolve_rate(self, attr_map, params)
    }
    fn cmt_raw(&self) -> usize {
        DoseEvent::cmt_raw(self)
    }
    fn bioavailable_infusion(&self, f_bio: f64) -> (f64, f64) {
        DoseEvent::bioavailable_infusion(self, f_bio)
    }
    fn with_bioavailable_infusion(&self, f_bio: f64) -> DoseEvent {
        DoseEvent::with_bioavailable_infusion(self, f_bio)
    }
    fn ss_cycle_doses(&self) -> Vec<DoseEvent> {
        DoseEvent::ss_cycle_doses(self)
    }
}
impl sealed::Sealed for DoseEvent {}

pub trait DoseAttrMapInternal: sealed::Sealed + Sized {
    fn zero_order_slot(&self, cmt: usize) -> Option<usize>;
    fn lowered_route_lag_slot(&self, cmt: usize) -> Option<usize>;
    fn lagtime_for_dose(&self, d: &DoseEvent, params: &[f64]) -> f64;
}
impl DoseAttrMapInternal for DoseAttrMap {
    fn zero_order_slot(&self, cmt: usize) -> Option<usize> {
        DoseAttrMap::zero_order_slot(self, cmt)
    }
    fn lowered_route_lag_slot(&self, cmt: usize) -> Option<usize> {
        DoseAttrMap::lowered_route_lag_slot(self, cmt)
    }
    fn lagtime_for_dose(&self, d: &DoseEvent, params: &[f64]) -> f64 {
        DoseAttrMap::lagtime_for_dose(self, d, params)
    }
}
impl sealed::Sealed for DoseAttrMap {}

pub trait SubjectInternal: sealed::Sealed + Sized {
    fn with_cmt_cutoffs(&self) -> std::borrow::Cow<'_, Subject>;
    fn controller_dose_off_at(&self, t: f64, cmt_idx: usize) -> f64;
    fn record_maps(&self) -> impl Iterator<Item = &HashMap<String, f64>>;
    fn n_scored_rows(&self) -> usize;
    fn record_covariate_snapshots(&self) -> Option<&[HashMap<String, f64>]>;
    #[cfg(feature = "survival")]
    fn retain_records(&mut self, keep: impl FnMut(&mut ObsRecord) -> bool);
    fn record_cov<'a>(
        &'a self, snaps: Option<&'a [HashMap<String, f64>]>, r: usize,
    ) -> &'a HashMap<String, f64>;
    fn param_time_obs(&self, j: usize) -> f64;
    fn param_time_pk_only(&self, m: usize) -> f64;
    fn param_time_at(&self, engine_t: f64) -> f64;
    fn dose_param_time(&self, k: usize) -> f64;
    fn implied_dose_parent(&self, k: usize) -> Option<usize>;
    fn same_instant(&self, j: usize, k: usize) -> bool;
    fn time_after_dose_at(&self, obs_t: f64) -> f64;
    fn time_after_dose_at_or_before(&self, j: usize) -> f64;
}
impl SubjectInternal for Subject {
    fn with_cmt_cutoffs(&self) -> std::borrow::Cow<'_, Subject> {
        Subject::with_cmt_cutoffs(self)
    }
    fn controller_dose_off_at(&self, t: f64, cmt_idx: usize) -> f64 {
        Subject::controller_dose_off_at(self, t, cmt_idx)
    }
    fn record_maps(&self) -> impl Iterator<Item = &HashMap<String, f64>> {
        Subject::record_maps(self)
    }
    fn n_scored_rows(&self) -> usize {
        Subject::n_scored_rows(self)
    }
    fn record_covariate_snapshots(&self) -> Option<&[HashMap<String, f64>]> {
        Subject::record_covariate_snapshots(self)
    }
    #[cfg(feature = "survival")]
    fn retain_records(&mut self, keep: impl FnMut(&mut ObsRecord) -> bool) {
        Subject::retain_records(self, keep)
    }
    fn record_cov<'a>(
        &'a self, snaps: Option<&'a [HashMap<String, f64>]>, r: usize,
    ) -> &'a HashMap<String, f64> {
        Subject::record_cov(self, snaps, r)
    }
    fn param_time_obs(&self, j: usize) -> f64 {
        Subject::param_time_obs(self, j)
    }
    fn param_time_pk_only(&self, m: usize) -> f64 {
        Subject::param_time_pk_only(self, m)
    }
    fn param_time_at(&self, engine_t: f64) -> f64 {
        Subject::param_time_at(self, engine_t)
    }
    fn dose_param_time(&self, k: usize) -> f64 {
        Subject::dose_param_time(self, k)
    }
    fn implied_dose_parent(&self, k: usize) -> Option<usize> {
        Subject::implied_dose_parent(self, k)
    }
    fn same_instant(&self, j: usize, k: usize) -> bool {
        Subject::same_instant(self, j, k)
    }
    fn time_after_dose_at(&self, obs_t: f64) -> f64 {
        Subject::time_after_dose_at(self, obs_t)
    }
    fn time_after_dose_at_or_before(&self, j: usize) -> f64 {
        Subject::time_after_dose_at_or_before(self, j)
    }
}
impl sealed::Sealed for Subject {}

pub trait OmegaMatrixInternal: sealed::Sealed + Sized {
    fn with_block_declared(self, block_declared: Vec<bool>) -> Self;
    fn structural_zeros(&self) -> Vec<bool>;
    fn sim_draw_factor<'a>(
        &'a self, declared: Option<&OmegaMatrix>, declared_fixed: &[bool],
    ) -> std::borrow::Cow<'a, DMatrix<f64>>;
}
impl OmegaMatrixInternal for OmegaMatrix {
    fn with_block_declared(self, block_declared: Vec<bool>) -> Self {
        OmegaMatrix::with_block_declared(self, block_declared)
    }
    fn structural_zeros(&self) -> Vec<bool> {
        OmegaMatrix::structural_zeros(self)
    }
    fn sim_draw_factor<'a>(
        &'a self, declared: Option<&OmegaMatrix>, declared_fixed: &[bool],
    ) -> std::borrow::Cow<'a, DMatrix<f64>> {
        OmegaMatrix::sim_draw_factor(self, declared, declared_fixed)
    }
}
impl sealed::Sealed for OmegaMatrix {}

pub trait PkModelInternal: sealed::Sealed + Sized {
    fn required_pk_params(&self) -> &'static [(usize, &'static str)];
    fn canonical_name(&self) -> &'static str;
    fn infusable_compartments(&self) -> &'static [usize];
    fn from_name(name: &str) -> Option<PkModel>;
    fn is_oral(&self) -> bool;
    fn consumes_pk_slot(&self, slot: usize) -> bool;
    fn topology(self) -> &'static PkTopology;
}
impl PkModelInternal for PkModel {
    fn required_pk_params(&self) -> &'static [(usize, &'static str)] {
        PkModel::required_pk_params(self)
    }
    fn canonical_name(&self) -> &'static str {
        PkModel::canonical_name(self)
    }
    fn infusable_compartments(&self) -> &'static [usize] {
        PkModel::infusable_compartments(self)
    }
    fn from_name(name: &str) -> Option<PkModel> {
        PkModel::from_name(name)
    }
    fn is_oral(&self) -> bool {
        PkModel::is_oral(self)
    }
    fn consumes_pk_slot(&self, slot: usize) -> bool {
        PkModel::consumes_pk_slot(self, slot)
    }
    fn topology(self) -> &'static PkTopology {
        PkModel::topology(self)
    }
}
impl sealed::Sealed for PkModel {}

pub trait RuvMagnitudeInternal: sealed::Sealed + Sized {
    fn has_endpoint_magnitude(&self) -> bool;
    fn eval_key(
        &self, key: usize, theta: &[f64], covariates: &HashMap<String, f64>, time: f64, tad: f64,
    ) -> Vec<f64>;
}
impl RuvMagnitudeInternal for RuvMagnitude {
    fn has_endpoint_magnitude(&self) -> bool {
        RuvMagnitude::has_endpoint_magnitude(self)
    }
    fn eval_key(
        &self, key: usize, theta: &[f64], covariates: &HashMap<String, f64>, time: f64, tad: f64,
    ) -> Vec<f64> {
        RuvMagnitude::eval_key(self, key, theta, covariates, time, tad)
    }
}
impl sealed::Sealed for RuvMagnitude {}

pub trait AnalyticReadoutInternal: sealed::Sealed + Sized {
    fn references_depot(&self) -> bool;
}
impl AnalyticReadoutInternal for AnalyticReadout {
    fn references_depot(&self) -> bool {
        AnalyticReadout::references_depot(self)
    }
}
impl sealed::Sealed for AnalyticReadout {}

pub trait PlanIdInternal: sealed::Sealed + Sized {
    fn get(&self) -> u64;
}
impl PlanIdInternal for PlanId {
    fn get(&self) -> u64 {
        PlanId::get(self)
    }
}
impl sealed::Sealed for PlanId {}

pub trait AbsorptionOdeEquivalentInternal: sealed::Sealed + Sized {
    fn built(&self) -> &CompiledModel;
}
impl AbsorptionOdeEquivalentInternal for AbsorptionOdeEquivalent {
    fn built(&self) -> &CompiledModel {
        AbsorptionOdeEquivalent::built(self)
    }
}
impl sealed::Sealed for AbsorptionOdeEquivalent {}

pub trait GradientMethodInternal: sealed::Sealed + Sized {
    fn forced_fd(model: &CompiledModel) -> bool;
    fn effective(model: &CompiledModel, options: &FitOptions) -> GradientMethod;
}
impl GradientMethodInternal for GradientMethod {
    fn forced_fd(model: &CompiledModel) -> bool {
        GradientMethod::forced_fd(model)
    }
    fn effective(model: &CompiledModel, options: &FitOptions) -> GradientMethod {
        GradientMethod::effective(model, options)
    }
}
impl sealed::Sealed for GradientMethod {}

pub trait CompiledModelInternal: sealed::Sealed + Sized {
    fn has_covariate_nn(&self) -> bool;
    fn nn_weight_theta_ranges(&self) -> Vec<std::ops::Range<usize>>;
    fn bound_from_fit(&self) -> bool;
    fn honors_ode_solver_opts(&self) -> bool;
    fn integrator_spec(&self) -> Option<&crate::ode::predictions::OdeSpec>;
    fn active_dose_attr_map(&self) -> &DoseAttrMap;
    fn pk_slot_rows(&self) -> Vec<(usize, usize)>;
    fn indiv_param_for_pk_slot(&self, pk_slot: usize) -> Option<usize>;
    fn has_dose_lag(&self) -> bool;
    fn has_route_absorption_lag(&self) -> bool;
    fn residual_cov_at<'a>(
        &'a self, params: &'a ModelParameters,
    ) -> crate::stats::residual_error::ResidualCov<'a>;
    fn residual_cov_theta<'a>(
        &'a self, theta: &[f64],
    ) -> crate::stats::residual_error::ResidualCov<'a>;
    fn residual_cov_with<'a>(
        &'a self, pairs: &'a [ResidualCorrelation], theta: &[f64],
    ) -> crate::stats::residual_error::ResidualCov<'a>;
    fn residual_is_dense(&self) -> bool;
    fn has_nested_levels(&self) -> bool;
    fn nested_levels(&self) -> Option<&NestedLevels>;
    fn derived_reads(&self) -> &[String];
    fn residual_variance_for_key(
        &self, key: usize, f_pred: f64, sigma: &[f64], correlations: &[ResidualCorrelation],
        ruv_scale: f64, mult: Option<&[f64]>,
    ) -> f64;
    fn bloq_in_force(&self) -> BloqMethod;
}
impl CompiledModelInternal for CompiledModel {
    fn has_covariate_nn(&self) -> bool {
        CompiledModel::has_covariate_nn(self)
    }
    fn nn_weight_theta_ranges(&self) -> Vec<std::ops::Range<usize>> {
        CompiledModel::nn_weight_theta_ranges(self)
    }
    fn bound_from_fit(&self) -> bool {
        CompiledModel::bound_from_fit(self)
    }
    fn honors_ode_solver_opts(&self) -> bool {
        CompiledModel::honors_ode_solver_opts(self)
    }
    fn integrator_spec(&self) -> Option<&crate::ode::predictions::OdeSpec> {
        CompiledModel::integrator_spec(self)
    }
    fn active_dose_attr_map(&self) -> &DoseAttrMap {
        CompiledModel::active_dose_attr_map(self)
    }
    fn pk_slot_rows(&self) -> Vec<(usize, usize)> {
        CompiledModel::pk_slot_rows(self)
    }
    fn indiv_param_for_pk_slot(&self, pk_slot: usize) -> Option<usize> {
        CompiledModel::indiv_param_for_pk_slot(self, pk_slot)
    }
    fn has_dose_lag(&self) -> bool {
        CompiledModel::has_dose_lag(self)
    }
    fn has_route_absorption_lag(&self) -> bool {
        CompiledModel::has_route_absorption_lag(self)
    }
    fn residual_cov_at<'a>(
        &'a self, params: &'a ModelParameters,
    ) -> crate::stats::residual_error::ResidualCov<'a> {
        CompiledModel::residual_cov_at(self, params)
    }
    fn residual_cov_theta<'a>(
        &'a self, theta: &[f64],
    ) -> crate::stats::residual_error::ResidualCov<'a> {
        CompiledModel::residual_cov_theta(self, theta)
    }
    fn residual_cov_with<'a>(
        &'a self, pairs: &'a [ResidualCorrelation], theta: &[f64],
    ) -> crate::stats::residual_error::ResidualCov<'a> {
        CompiledModel::residual_cov_with(self, pairs, theta)
    }
    fn residual_is_dense(&self) -> bool {
        CompiledModel::residual_is_dense(self)
    }
    fn has_nested_levels(&self) -> bool {
        CompiledModel::has_nested_levels(self)
    }
    fn nested_levels(&self) -> Option<&NestedLevels> {
        CompiledModel::nested_levels(self)
    }
    fn derived_reads(&self) -> &[String] {
        CompiledModel::derived_reads(self)
    }
    fn residual_variance_for_key(
        &self, key: usize, f_pred: f64, sigma: &[f64], correlations: &[ResidualCorrelation],
        ruv_scale: f64, mult: Option<&[f64]>,
    ) -> f64 {
        CompiledModel::residual_variance_for_key(
            self,
            key,
            f_pred,
            sigma,
            correlations,
            ruv_scale,
            mult,
        )
    }
    fn bloq_in_force(&self) -> BloqMethod {
        CompiledModel::bloq_in_force(self)
    }
}
impl sealed::Sealed for CompiledModel {}

pub trait BloqMethodInternal: sealed::Sealed + Sized {
    fn from_label(label: &str) -> Option<Self>;
}
impl BloqMethodInternal for BloqMethod {
    fn from_label(label: &str) -> Option<Self> {
        BloqMethod::from_label(label)
    }
}
impl sealed::Sealed for BloqMethod {}

pub trait OptimizerInternal: sealed::Sealed + Sized {
    fn resolve_auto_given_analytic(
        self, model: &CompiledModel, analytic_outer_gradient: bool,
    ) -> Optimizer;
}
impl OptimizerInternal for Optimizer {
    fn resolve_auto_given_analytic(
        self, model: &CompiledModel, analytic_outer_gradient: bool,
    ) -> Optimizer {
        Optimizer::resolve_auto_given_analytic(self, model, analytic_outer_gradient)
    }
}
impl sealed::Sealed for Optimizer {}

pub trait EstimationMethodInternal: sealed::Sealed + Sized {
    fn evaluates_at_maxiter_zero(self) -> bool;
    fn requires_random_effects(self) -> bool;
}
impl EstimationMethodInternal for EstimationMethod {
    fn evaluates_at_maxiter_zero(self) -> bool {
        EstimationMethod::evaluates_at_maxiter_zero(self)
    }
    fn requires_random_effects(self) -> bool {
        EstimationMethod::requires_random_effects(self)
    }
}
impl sealed::Sealed for EstimationMethod {}

pub trait FitOptionsInternal: sealed::Sealed + Sized {
    fn inner_settings(&self) -> crate::estimation::inner_settings::InnerSettings;
    fn forces_fd(&self) -> bool;
    fn ode_solver_override(&self) -> crate::ode::solver::OdeSolverOverride;
    fn eval_only_methods(&self) -> Vec<EstimationMethod>;
    fn evaluation_only_chain(&self) -> bool;
    fn evaluation_only_run(&self) -> bool;
    fn covariance_stage(&self) -> Option<EstimationMethod>;
    fn agq_nodes_for(method: EstimationMethod, n_agq: usize) -> Option<usize>;
    fn unsupported_keys_warnings_with_model(&self, model: &CompiledModel) -> Vec<String>;
}
impl FitOptionsInternal for FitOptions {
    fn inner_settings(&self) -> crate::estimation::inner_settings::InnerSettings {
        FitOptions::inner_settings(self)
    }
    fn forces_fd(&self) -> bool {
        FitOptions::forces_fd(self)
    }
    fn ode_solver_override(&self) -> crate::ode::solver::OdeSolverOverride {
        FitOptions::ode_solver_override(self)
    }
    fn eval_only_methods(&self) -> Vec<EstimationMethod> {
        FitOptions::eval_only_methods(self)
    }
    fn evaluation_only_chain(&self) -> bool {
        FitOptions::evaluation_only_chain(self)
    }
    fn evaluation_only_run(&self) -> bool {
        FitOptions::evaluation_only_run(self)
    }
    fn covariance_stage(&self) -> Option<EstimationMethod> {
        FitOptions::covariance_stage(self)
    }
    fn agq_nodes_for(method: EstimationMethod, n_agq: usize) -> Option<usize> {
        FitOptions::agq_nodes_for(method, n_agq)
    }
    fn unsupported_keys_warnings_with_model(&self, model: &CompiledModel) -> Vec<String> {
        FitOptions::unsupported_keys_warnings_with_model(self, model)
    }
}
impl sealed::Sealed for FitOptions {}

pub trait Dual2Internal<const N: usize>: sealed::Sealed + Sized {
    fn same_bits(&self, o: &Self) -> bool;
}
impl<const N: usize> Dual2Internal<N> for Dual2<N> {
    fn same_bits(&self, o: &Self) -> bool {
        Dual2::same_bits(self, o)
    }
}
impl<const N: usize> sealed::Sealed for Dual2<N> {}

pub trait LevelContrastInternal: sealed::Sealed + Sized {
    fn label(self) -> &'static str;
}
impl LevelContrastInternal for LevelContrast {
    fn label(self) -> &'static str {
        LevelContrast::label(self)
    }
}
impl sealed::Sealed for LevelContrast {}

pub trait ThetaBlocksInternal: sealed::Sealed + Sized {
    fn profiled_names(&self) -> Vec<String>;
    fn profiled_theta_indices(&self) -> Vec<usize>;
}
impl ThetaBlocksInternal for ThetaBlocks {
    fn profiled_names(&self) -> Vec<String> {
        ThetaBlocks::profiled_names(self)
    }
    fn profiled_theta_indices(&self) -> Vec<usize> {
        ThetaBlocks::profiled_theta_indices(self)
    }
}
impl sealed::Sealed for ThetaBlocks {}

pub trait BytecodeInternal: sealed::Sealed + Sized {
    fn push_const(&mut self, v: f64);
}
impl BytecodeInternal for Bytecode {
    fn push_const(&mut self, v: f64) {
        Bytecode::push_const(self, v)
    }
}
impl sealed::Sealed for Bytecode {}

pub trait OdeRhsProgramInternal: sealed::Sealed + Sized {
    fn route_note(&self) -> Option<(&'static str, String, serde_json::Value)>;
    fn init_state_g<T: crate::sens::num::vm::VmNum>(
        &self, params: &[T], n_states: usize,
    ) -> Option<Vec<T>>;
    fn uses_time_vars(&self) -> bool;
    fn reads_time_builtin(&self) -> bool;
    fn reads_model_time(&self) -> bool;
    #[cfg(feature = "survival")]
    fn chz_dependency_slice(&self, chz_slot: usize) -> Vec<bool>;
    #[cfg(feature = "survival")]
    fn indiv_param_reaches_chz(&self, i: usize, slice: &[bool]) -> bool;
    #[cfg(feature = "survival")]
    fn init_reads_indiv_param_reaching_chz(&self, i: usize, slice: &[bool]) -> bool;
    #[cfg(feature = "survival")]
    fn has_indiv_param(&self, i: usize) -> bool;
    fn pk_reads_model_time(&self) -> bool;
    fn pk_reads_tafd(&self) -> bool;
    fn pk_reads_solver_time(&self) -> bool;
    fn pk_reads_absolute_time(&self) -> bool;
    fn eval_rhs_g<T: crate::sens::num::vm::VmNum>(
        &self, u: &[T], params: &[T], t: f64, tafd: f64, tad: T, dt: T, du: &mut [T],
        vars: &mut Vec<T>, stack: &mut Vec<T>,
    );
}
impl OdeRhsProgramInternal for OdeRhsProgram {
    fn route_note(&self) -> Option<(&'static str, String, serde_json::Value)> {
        OdeRhsProgram::route_note(self)
    }
    fn init_state_g<T: crate::sens::num::vm::VmNum>(
        &self, params: &[T], n_states: usize,
    ) -> Option<Vec<T>> {
        OdeRhsProgram::init_state_g(self, params, n_states)
    }
    fn uses_time_vars(&self) -> bool {
        OdeRhsProgram::uses_time_vars(self)
    }
    fn reads_time_builtin(&self) -> bool {
        OdeRhsProgram::reads_time_builtin(self)
    }
    fn reads_model_time(&self) -> bool {
        OdeRhsProgram::reads_model_time(self)
    }
    #[cfg(feature = "survival")]
    fn chz_dependency_slice(&self, chz_slot: usize) -> Vec<bool> {
        OdeRhsProgram::chz_dependency_slice(self, chz_slot)
    }
    #[cfg(feature = "survival")]
    fn indiv_param_reaches_chz(&self, i: usize, slice: &[bool]) -> bool {
        OdeRhsProgram::indiv_param_reaches_chz(self, i, slice)
    }
    #[cfg(feature = "survival")]
    fn init_reads_indiv_param_reaching_chz(&self, i: usize, slice: &[bool]) -> bool {
        OdeRhsProgram::init_reads_indiv_param_reaching_chz(self, i, slice)
    }
    #[cfg(feature = "survival")]
    fn has_indiv_param(&self, i: usize) -> bool {
        OdeRhsProgram::has_indiv_param(self, i)
    }
    fn pk_reads_model_time(&self) -> bool {
        OdeRhsProgram::pk_reads_model_time(self)
    }
    fn pk_reads_tafd(&self) -> bool {
        OdeRhsProgram::pk_reads_tafd(self)
    }
    fn pk_reads_solver_time(&self) -> bool {
        OdeRhsProgram::pk_reads_solver_time(self)
    }
    fn pk_reads_absolute_time(&self) -> bool {
        OdeRhsProgram::pk_reads_absolute_time(self)
    }
    fn eval_rhs_g<T: crate::sens::num::vm::VmNum>(
        &self, u: &[T], params: &[T], t: f64, tafd: f64, tad: T, dt: T, du: &mut [T],
        vars: &mut Vec<T>, stack: &mut Vec<T>,
    ) {
        OdeRhsProgram::eval_rhs_g(self, u, params, t, tafd, tad, dt, du, vars, stack)
    }
}
impl sealed::Sealed for OdeRhsProgram {}

#[cfg(feature = "markov")]
pub trait CtmmGeneratorProgramInternal: sealed::Sealed + Sized {
    #[cfg(feature = "markov")]
    fn eval_generator_colmajor_g<T: crate::sens::num::vm::VmNum>(
        &self, theta: &[T], eta: &[T], covariates: &HashMap<String, f64>,
    ) -> Option<Vec<T>>;
}
#[cfg(feature = "markov")]
impl CtmmGeneratorProgramInternal for CtmmGeneratorProgram {
    #[cfg(feature = "markov")]
    fn eval_generator_colmajor_g<T: crate::sens::num::vm::VmNum>(
        &self, theta: &[T], eta: &[T], covariates: &HashMap<String, f64>,
    ) -> Option<Vec<T>> {
        CtmmGeneratorProgram::eval_generator_colmajor_g(self, theta, eta, covariates)
    }
}
#[cfg(feature = "markov")]
impl sealed::Sealed for CtmmGeneratorProgram {}

#[cfg(feature = "survival")]
pub trait EndpointExprProgramInternal: sealed::Sealed + Sized {
    #[cfg(feature = "survival")]
    fn eval_g<T: crate::sens::num::vm::VmNum>(
        &self, theta: &[T], eta: &[T], covariates: &HashMap<String, f64>,
    ) -> Option<Vec<T>>;
}
#[cfg(feature = "survival")]
impl EndpointExprProgramInternal for EndpointExprProgram {
    #[cfg(feature = "survival")]
    fn eval_g<T: crate::sens::num::vm::VmNum>(
        &self, theta: &[T], eta: &[T], covariates: &HashMap<String, f64>,
    ) -> Option<Vec<T>> {
        EndpointExprProgram::eval_g(self, theta, eta, covariates)
    }
}
#[cfg(feature = "survival")]
impl sealed::Sealed for EndpointExprProgram {}

pub trait IndivParamProgramInternal: sealed::Sealed + Sized {
    fn n_eta_axis(&self) -> usize;
    fn pk_slots_ref(&self) -> &[usize];
    #[cfg(feature = "survival")]
    fn params_reading_time(&self) -> Vec<bool>;
    #[cfg(feature = "markov")]
    fn params_reading_covariates(&self, names: &[&str]) -> Vec<bool>;
    fn pk_var_slots(&self) -> &[(usize, usize)];
    fn eval_params_g<T: crate::sens::num::vm::VmNum>(
        &self, theta: &[T], eta: &[T], covariates: &HashMap<String, f64>, nn: &[Vec<T>],
    ) -> Vec<T>;
    fn eval_param_values(
        &self, theta: &[f64], eta: &[f64], covariates: &HashMap<String, f64>, time: f64,
        nn_outputs: &[Vec<f64>],
    ) -> Vec<f64>;
}
impl IndivParamProgramInternal for IndivParamProgram {
    fn n_eta_axis(&self) -> usize {
        IndivParamProgram::n_eta_axis(self)
    }
    fn pk_slots_ref(&self) -> &[usize] {
        IndivParamProgram::pk_slots_ref(self)
    }
    #[cfg(feature = "survival")]
    fn params_reading_time(&self) -> Vec<bool> {
        IndivParamProgram::params_reading_time(self)
    }
    #[cfg(feature = "markov")]
    fn params_reading_covariates(&self, names: &[&str]) -> Vec<bool> {
        IndivParamProgram::params_reading_covariates(self, names)
    }
    fn pk_var_slots(&self) -> &[(usize, usize)] {
        IndivParamProgram::pk_var_slots(self)
    }
    fn eval_params_g<T: crate::sens::num::vm::VmNum>(
        &self, theta: &[T], eta: &[T], covariates: &HashMap<String, f64>, nn: &[Vec<T>],
    ) -> Vec<T> {
        IndivParamProgram::eval_params_g(self, theta, eta, covariates, nn)
    }
    fn eval_param_values(
        &self, theta: &[f64], eta: &[f64], covariates: &HashMap<String, f64>, time: f64,
        nn_outputs: &[Vec<f64>],
    ) -> Vec<f64> {
        IndivParamProgram::eval_param_values(self, theta, eta, covariates, time, nn_outputs)
    }
}
impl sealed::Sealed for IndivParamProgram {}

pub trait OdeOutputProgramInternal: sealed::Sealed + Sized {
    fn reads_leaves(&self) -> bool;
    fn reads_time_builtin(&self) -> bool;
}
impl OdeOutputProgramInternal for OdeOutputProgram {
    fn reads_leaves(&self) -> bool {
        OdeOutputProgram::reads_leaves(self)
    }
    fn reads_time_builtin(&self) -> bool {
        OdeOutputProgram::reads_time_builtin(self)
    }
}
impl sealed::Sealed for OdeOutputProgram {}

pub trait ScaleDerivProgramInternal: sealed::Sealed + Sized {
    fn n_eta_axis(&self) -> usize;
}
impl ScaleDerivProgramInternal for ScaleDerivProgram {
    fn n_eta_axis(&self) -> usize {
        ScaleDerivProgram::n_eta_axis(self)
    }
}
impl sealed::Sealed for ScaleDerivProgram {}

pub trait RuvMagDerivProgramInternal: sealed::Sealed + Sized {
    fn eval_g<T: crate::sens::num::vm::VmNum>(
        &self, theta: &[T], cov: &HashMap<String, f64>, time: f64, tad: f64,
    ) -> Option<T>;
}
impl RuvMagDerivProgramInternal for RuvMagDerivProgram {
    fn eval_g<T: crate::sens::num::vm::VmNum>(
        &self, theta: &[T], cov: &HashMap<String, f64>, time: f64, tad: f64,
    ) -> Option<T> {
        RuvMagDerivProgram::eval_g(self, theta, cov, time, tad)
    }
}
impl sealed::Sealed for RuvMagDerivProgram {}

pub trait ErrorSpecInternal: sealed::Sealed + Sized {
    fn slot_reads_f(&self, cmt: usize, k: usize, n_sigma: usize) -> bool;
    fn variance_at_raw(&self, cmt: usize, f_pred: f64, sigma: &[f64]) -> f64;
}
impl ErrorSpecInternal for ErrorSpec {
    fn slot_reads_f(&self, cmt: usize, k: usize, n_sigma: usize) -> bool {
        ErrorSpec::slot_reads_f(self, cmt, k, n_sigma)
    }
    fn variance_at_raw(&self, cmt: usize, f_pred: f64, sigma: &[f64]) -> f64 {
        ErrorSpec::variance_at_raw(self, cmt, f_pred, sigma)
    }
}
impl sealed::Sealed for ErrorSpec {}

pub trait OdeSolverStatsInternal: sealed::Sealed + Sized {
    fn kept_clamped_steps(&self) -> usize;
    fn kept_unfinished_segments(&self) -> usize;
    fn merge(&mut self, other: &OdeSolverStats);
}
impl OdeSolverStatsInternal for OdeSolverStats {
    fn kept_clamped_steps(&self) -> usize {
        OdeSolverStats::kept_clamped_steps(self)
    }
    fn kept_unfinished_segments(&self) -> usize {
        OdeSolverStats::kept_unfinished_segments(self)
    }
    fn merge(&mut self, other: &OdeSolverStats) {
        OdeSolverStats::merge(self, other)
    }
}
impl sealed::Sealed for OdeSolverStats {}

pub trait OdeSpecInternal: sealed::Sealed + Sized {
    fn effective_solver_opts(&self) -> crate::ode::OdeSolverOptions;
}
impl OdeSpecInternal for OdeSpec {
    fn effective_solver_opts(&self) -> crate::ode::OdeSolverOptions {
        OdeSpec::effective_solver_opts(self)
    }
}
impl sealed::Sealed for OdeSpec {}

pub trait AdaptiveRuleInternal: sealed::Sealed + Sized {
    fn matches(&self, signal: f64) -> bool;
    fn label(&self) -> String;
}
impl AdaptiveRuleInternal for AdaptiveRule {
    fn matches(&self, signal: f64) -> bool {
        AdaptiveRule::matches(self, signal)
    }
    fn label(&self) -> String {
        AdaptiveRule::label(self)
    }
}
impl sealed::Sealed for AdaptiveRule {}

pub trait AdaptiveDosingSpecInternal: sealed::Sealed + Sized {
    fn literal_dose_plan(&self) -> Option<SubjectDosePlan>;
    fn dose_grid(&self, bounds: (f64, f64)) -> Option<DoseGrid>;
    fn smallest_increase_percent(&self) -> Option<f64>;
    fn absorbed_increase(&self, plan: &SubjectDosePlan) -> Option<AbsorbedIncrease>;
    fn dose_expressions(&self) -> Vec<(&'static str, &str)>;
}
impl AdaptiveDosingSpecInternal for AdaptiveDosingSpec {
    fn literal_dose_plan(&self) -> Option<SubjectDosePlan> {
        AdaptiveDosingSpec::literal_dose_plan(self)
    }
    fn dose_grid(&self, bounds: (f64, f64)) -> Option<DoseGrid> {
        AdaptiveDosingSpec::dose_grid(self, bounds)
    }
    fn smallest_increase_percent(&self) -> Option<f64> {
        AdaptiveDosingSpec::smallest_increase_percent(self)
    }
    fn absorbed_increase(&self, plan: &SubjectDosePlan) -> Option<AbsorbedIncrease> {
        AdaptiveDosingSpec::absorbed_increase(self, plan)
    }
    fn dose_expressions(&self) -> Vec<(&'static str, &str)> {
        AdaptiveDosingSpec::dose_expressions(self)
    }
}
impl sealed::Sealed for AdaptiveDosingSpec {}

pub trait IntervalStepInternal: sealed::Sealed + Sized {
    fn label(self) -> &'static str;
}
impl IntervalStepInternal for IntervalStep {
    fn label(self) -> &'static str {
        IntervalStep::label(self)
    }
}
impl sealed::Sealed for IntervalStep {}

pub trait AdaptiveActionInternal: sealed::Sealed + Sized {
    fn label(&self) -> String;
}
impl AdaptiveActionInternal for AdaptiveAction {
    fn label(&self) -> String {
        AdaptiveAction::label(self)
    }
}
impl sealed::Sealed for AdaptiveAction {}

pub trait PopulationFingerprintInternal: sealed::Sealed + Sized {
    fn occasions_derived(&self) -> bool;
    fn of_with(population: &Population, occasions_derived: bool) -> Self;
    fn is_current(&self) -> bool;
    fn first_subject_difference(&self, population: &Population) -> Option<Difference>;
    fn first_difference(&self, population: &Population) -> Option<Difference>;
}
impl PopulationFingerprintInternal for PopulationFingerprint {
    fn occasions_derived(&self) -> bool {
        PopulationFingerprint::occasions_derived(self)
    }
    fn of_with(population: &Population, occasions_derived: bool) -> Self {
        PopulationFingerprint::of_with(population, occasions_derived)
    }
    fn is_current(&self) -> bool {
        PopulationFingerprint::is_current(self)
    }
    fn first_subject_difference(&self, population: &Population) -> Option<Difference> {
        PopulationFingerprint::first_subject_difference(self, population)
    }
    fn first_difference(&self, population: &Population) -> Option<Difference> {
        PopulationFingerprint::first_difference(self, population)
    }
}
impl sealed::Sealed for PopulationFingerprint {}

pub trait ScoringSettingsInternal: sealed::Sealed + Sized {
    fn of_run(model: &CompiledModel, options: &FitOptions) -> Self;
    fn overwrite(&self, options: &FitOptions) -> FitOptions;
    fn check_domain(&self) -> Result<(), OutOfDomain>;
}
impl ScoringSettingsInternal for ScoringSettings {
    fn of_run(model: &CompiledModel, options: &FitOptions) -> Self {
        ScoringSettings::of_run(model, options)
    }
    fn overwrite(&self, options: &FitOptions) -> FitOptions {
        ScoringSettings::overwrite(self, options)
    }
    fn check_domain(&self) -> Result<(), OutOfDomain> {
        ScoringSettings::check_domain(self)
    }
}
impl sealed::Sealed for ScoringSettings {}

pub trait SirSettingsInternal: sealed::Sealed + Sized {
    fn check_domain(&self) -> Result<(), crate::estimation::population_check::OutOfDomain>;
}
impl SirSettingsInternal for SirSettings {
    fn check_domain(&self) -> Result<(), crate::estimation::population_check::OutOfDomain> {
        SirSettings::check_domain(self)
    }
}
impl sealed::Sealed for SirSettings {}

pub trait SelectionFilterInternal: sealed::Sealed + Sized {
    fn references_column(&self, col: &str) -> bool;
}
impl SelectionFilterInternal for SelectionFilter {
    fn references_column(&self, col: &str) -> bool {
        SelectionFilter::references_column(self, col)
    }
}
impl sealed::Sealed for SelectionFilter {}

pub trait CovariateMuRefFields: sealed::Sealed + Sized {
    fn colliding_anchors(&self) -> &Vec<CovariateMuRef>;
    fn colliding_anchors_mut(&mut self) -> &mut Vec<CovariateMuRef>;
    fn eta_shared(&self) -> bool;
    fn eta_shared_mut(&mut self) -> &mut bool;
    fn read_outside_groups(&self) -> &Vec<String>;
    fn read_outside_groups_mut(&mut self) -> &mut Vec<String>;
    fn read_outside_groups_and_anchors(&self) -> &Vec<String>;
    fn read_outside_groups_and_anchors_mut(&mut self) -> &mut Vec<String>;
    fn typical(&self) -> &crate::parser::model_parser::TypicalProgram;
    fn typical_mut(&mut self) -> &mut crate::parser::model_parser::TypicalProgram;
}
impl CovariateMuRefFields for CovariateMuRef {
    fn colliding_anchors(&self) -> &Vec<CovariateMuRef> {
        &self.colliding_anchors
    }
    fn colliding_anchors_mut(&mut self) -> &mut Vec<CovariateMuRef> {
        &mut self.colliding_anchors
    }
    fn eta_shared(&self) -> bool {
        self.eta_shared
    }
    fn eta_shared_mut(&mut self) -> &mut bool {
        &mut self.eta_shared
    }
    fn read_outside_groups(&self) -> &Vec<String> {
        &self.read_outside_groups
    }
    fn read_outside_groups_mut(&mut self) -> &mut Vec<String> {
        &mut self.read_outside_groups
    }
    fn read_outside_groups_and_anchors(&self) -> &Vec<String> {
        &self.read_outside_groups_and_anchors
    }
    fn read_outside_groups_and_anchors_mut(&mut self) -> &mut Vec<String> {
        &mut self.read_outside_groups_and_anchors
    }
    fn typical(&self) -> &crate::parser::model_parser::TypicalProgram {
        &self.typical
    }
    fn typical_mut(&mut self) -> &mut crate::parser::model_parser::TypicalProgram {
        &mut self.typical
    }
}
impl sealed::Sealed for CovariateMuRef {}

pub trait DoseEventFields: sealed::Sealed + Sized {
    fn cmt(&self) -> usize;
    fn cmt_mut(&mut self) -> &mut usize;
    fn evid4(&self) -> bool;
    fn evid4_mut(&mut self) -> &mut bool;
    fn implied(&self) -> &Option<ImpliedDose>;
    fn implied_mut(&mut self) -> &mut Option<ImpliedDose>;
    fn off_at(&self) -> f64;
    fn off_at_mut(&mut self) -> &mut f64;
}
impl DoseEventFields for DoseEvent {
    fn cmt(&self) -> usize {
        self.cmt
    }
    fn cmt_mut(&mut self) -> &mut usize {
        &mut self.cmt
    }
    fn evid4(&self) -> bool {
        self.evid4
    }
    fn evid4_mut(&mut self) -> &mut bool {
        &mut self.evid4
    }
    fn implied(&self) -> &Option<ImpliedDose> {
        &self.implied
    }
    fn implied_mut(&mut self) -> &mut Option<ImpliedDose> {
        &mut self.implied
    }
    fn off_at(&self) -> f64 {
        self.off_at
    }
    fn off_at_mut(&mut self) -> &mut f64 {
        &mut self.off_at
    }
}

pub trait IndivParamPartialsFields: sealed::Sealed + Sized {
    fn d_d_eta(&self) -> &Vec<Vec<Expression>>;
    fn d_d_eta_mut(&mut self) -> &mut Vec<Vec<Expression>>;
    fn d_d_theta(&self) -> &Vec<Vec<Expression>>;
    fn d_d_theta_mut(&mut self) -> &mut Vec<Vec<Expression>>;
    fn derived_reads(&self) -> &Vec<String>;
    fn derived_reads_mut(&mut self) -> &mut Vec<String>;
    fn indiv_param_program(&self) -> &Option<IndivParamProgram>;
    fn indiv_param_program_mut(&mut self) -> &mut Option<IndivParamProgram>;
    fn lognormal_eta_chain(&self) -> Option<bool>;
    fn lognormal_eta_chain_mut(&mut self) -> &mut Option<bool>;
    fn mu_fold(&self) -> &MuFoldSpec;
    fn mu_fold_mut(&mut self) -> &mut MuFoldSpec;
    fn mu_ref_owners(&self) -> &std::collections::BTreeMap<String, String>;
    fn mu_ref_owners_mut(&mut self) -> &mut std::collections::BTreeMap<String, String>;
    fn names(&self) -> &Vec<String>;
    fn names_mut(&mut self) -> &mut Vec<String>;
    fn nested_levels(&self) -> &Option<crate::types::NestedLevels>;
    fn nested_levels_mut(&mut self) -> &mut Option<crate::types::NestedLevels>;
    fn slots(&self) -> &Vec<usize>;
    fn slots_mut(&mut self) -> &mut Vec<usize>;
    fn tangent_base(&self) -> usize;
    fn tangent_base_mut(&mut self) -> &mut usize;
    fn theta_blocks(&self) -> &ThetaBlocks;
    fn theta_blocks_mut(&mut self) -> &mut ThetaBlocks;
    fn values(&self) -> &Vec<Expression>;
    fn values_mut(&mut self) -> &mut Vec<Expression>;
}
impl IndivParamPartialsFields for IndivParamPartials {
    fn d_d_eta(&self) -> &Vec<Vec<Expression>> {
        &self.d_d_eta
    }
    fn d_d_eta_mut(&mut self) -> &mut Vec<Vec<Expression>> {
        &mut self.d_d_eta
    }
    fn d_d_theta(&self) -> &Vec<Vec<Expression>> {
        &self.d_d_theta
    }
    fn d_d_theta_mut(&mut self) -> &mut Vec<Vec<Expression>> {
        &mut self.d_d_theta
    }
    fn derived_reads(&self) -> &Vec<String> {
        &self.derived_reads
    }
    fn derived_reads_mut(&mut self) -> &mut Vec<String> {
        &mut self.derived_reads
    }
    fn indiv_param_program(&self) -> &Option<IndivParamProgram> {
        &self.indiv_param_program
    }
    fn indiv_param_program_mut(&mut self) -> &mut Option<IndivParamProgram> {
        &mut self.indiv_param_program
    }
    fn lognormal_eta_chain(&self) -> Option<bool> {
        self.lognormal_eta_chain
    }
    fn lognormal_eta_chain_mut(&mut self) -> &mut Option<bool> {
        &mut self.lognormal_eta_chain
    }
    fn mu_fold(&self) -> &MuFoldSpec {
        &self.mu_fold
    }
    fn mu_fold_mut(&mut self) -> &mut MuFoldSpec {
        &mut self.mu_fold
    }
    fn mu_ref_owners(&self) -> &std::collections::BTreeMap<String, String> {
        &self.mu_ref_owners
    }
    fn mu_ref_owners_mut(&mut self) -> &mut std::collections::BTreeMap<String, String> {
        &mut self.mu_ref_owners
    }
    fn names(&self) -> &Vec<String> {
        &self.names
    }
    fn names_mut(&mut self) -> &mut Vec<String> {
        &mut self.names
    }
    fn nested_levels(&self) -> &Option<crate::types::NestedLevels> {
        &self.nested_levels
    }
    fn nested_levels_mut(&mut self) -> &mut Option<crate::types::NestedLevels> {
        &mut self.nested_levels
    }
    fn slots(&self) -> &Vec<usize> {
        &self.slots
    }
    fn slots_mut(&mut self) -> &mut Vec<usize> {
        &mut self.slots
    }
    fn tangent_base(&self) -> usize {
        self.tangent_base
    }
    fn tangent_base_mut(&mut self) -> &mut usize {
        &mut self.tangent_base
    }
    fn theta_blocks(&self) -> &ThetaBlocks {
        &self.theta_blocks
    }
    fn theta_blocks_mut(&mut self) -> &mut ThetaBlocks {
        &mut self.theta_blocks
    }
    fn values(&self) -> &Vec<Expression> {
        &self.values
    }
    fn values_mut(&mut self) -> &mut Vec<Expression> {
        &mut self.values
    }
}
impl sealed::Sealed for IndivParamPartials {}

pub trait IndivParamProgramFields: sealed::Sealed + Sized {
    fn cov_static_mask(&self) -> &Vec<bool>;
    fn cov_static_mask_mut(&mut self) -> &mut Vec<bool>;
    fn n_vars(&self) -> usize;
    fn n_vars_mut(&mut self) -> &mut usize;
    fn stmts(&self) -> &Vec<Statement>;
    fn stmts_mut(&mut self) -> &mut Vec<Statement>;
}
impl IndivParamProgramFields for IndivParamProgram {
    fn cov_static_mask(&self) -> &Vec<bool> {
        &self.cov_static_mask
    }
    fn cov_static_mask_mut(&mut self) -> &mut Vec<bool> {
        &mut self.cov_static_mask
    }
    fn n_vars(&self) -> usize {
        self.n_vars
    }
    fn n_vars_mut(&mut self) -> &mut usize {
        &mut self.n_vars
    }
    fn stmts(&self) -> &Vec<Statement> {
        &self.stmts
    }
    fn stmts_mut(&mut self) -> &mut Vec<Statement> {
        &mut self.stmts
    }
}

pub trait LevelBlockDeclFields: sealed::Sealed + Sized {
    fn eta_couplings(&self) -> &Vec<EtaCoupling>;
    fn eta_couplings_mut(&mut self) -> &mut Vec<EtaCoupling>;
    fn level_capable_cmts(&self) -> &Vec<usize>;
    fn level_capable_cmts_mut(&mut self) -> &mut Vec<usize>;
    fn read_bare(&self) -> bool;
    fn read_bare_mut(&mut self) -> &mut bool;
    fn record_cmts(&self) -> &Vec<usize>;
    fn record_cmts_mut(&mut self) -> &mut Vec<usize>;
}
impl LevelBlockDeclFields for LevelBlockDecl {
    fn eta_couplings(&self) -> &Vec<EtaCoupling> {
        &self.eta_couplings
    }
    fn eta_couplings_mut(&mut self) -> &mut Vec<EtaCoupling> {
        &mut self.eta_couplings
    }
    fn level_capable_cmts(&self) -> &Vec<usize> {
        &self.level_capable_cmts
    }
    fn level_capable_cmts_mut(&mut self) -> &mut Vec<usize> {
        &mut self.level_capable_cmts
    }
    fn read_bare(&self) -> bool {
        self.read_bare
    }
    fn read_bare_mut(&mut self) -> &mut bool {
        &mut self.read_bare
    }
    fn record_cmts(&self) -> &Vec<usize> {
        &self.record_cmts
    }
    fn record_cmts_mut(&mut self) -> &mut Vec<usize> {
        &mut self.record_cmts
    }
}
impl sealed::Sealed for LevelBlockDecl {}

pub trait OdeOutputProgramFields: sealed::Sealed + Sized {
    fn bc(&self) -> &Bytecode;
    fn bc_mut(&mut self) -> &mut Bytecode;
}
impl OdeOutputProgramFields for OdeOutputProgram {
    fn bc(&self) -> &Bytecode {
        &self.bc
    }
    fn bc_mut(&mut self) -> &mut Bytecode {
        &mut self.bc
    }
}

pub trait OdeRhsProgramFields: sealed::Sealed + Sized {
    fn binding(&self) -> &RhsBindingPlan;
    fn binding_mut(&mut self) -> &mut RhsBindingPlan;
    fn hoist(&self) -> &Option<RhsHoist>;
    fn hoist_mut(&mut self) -> &mut Option<RhsHoist>;
    fn indiv_to_params_slot(&self) -> &Vec<usize>;
    fn indiv_to_params_slot_mut(&mut self) -> &mut Vec<usize>;
    fn macheps_slot(&self) -> usize;
    fn macheps_slot_mut(&mut self) -> &mut usize;
    fn n_vars_total(&self) -> usize;
    fn n_vars_total_mut(&mut self) -> &mut usize;
    fn reg(&self) -> &Option<crate::parser::model_parser::rhs_regcode::RhsRegProgram>;
    fn reg_mut(&mut self) -> &mut Option<crate::parser::model_parser::rhs_regcode::RhsRegProgram>;
    fn stmts(&self) -> &Vec<Statement>;
    fn stmts_mut(&mut self) -> &mut Vec<Statement>;
}
impl OdeRhsProgramFields for OdeRhsProgram {
    fn binding(&self) -> &RhsBindingPlan {
        &self.binding
    }
    fn binding_mut(&mut self) -> &mut RhsBindingPlan {
        &mut self.binding
    }
    fn hoist(&self) -> &Option<RhsHoist> {
        &self.hoist
    }
    fn hoist_mut(&mut self) -> &mut Option<RhsHoist> {
        &mut self.hoist
    }
    fn indiv_to_params_slot(&self) -> &Vec<usize> {
        &self.indiv_to_params_slot
    }
    fn indiv_to_params_slot_mut(&mut self) -> &mut Vec<usize> {
        &mut self.indiv_to_params_slot
    }
    fn macheps_slot(&self) -> usize {
        self.macheps_slot
    }
    fn macheps_slot_mut(&mut self) -> &mut usize {
        &mut self.macheps_slot
    }
    fn n_vars_total(&self) -> usize {
        self.n_vars_total
    }
    fn n_vars_total_mut(&mut self) -> &mut usize {
        &mut self.n_vars_total
    }
    fn reg(&self) -> &Option<crate::parser::model_parser::rhs_regcode::RhsRegProgram> {
        &self.reg
    }
    fn reg_mut(&mut self) -> &mut Option<crate::parser::model_parser::rhs_regcode::RhsRegProgram> {
        &mut self.reg
    }
    fn stmts(&self) -> &Vec<Statement> {
        &self.stmts
    }
    fn stmts_mut(&mut self) -> &mut Vec<Statement> {
        &mut self.stmts
    }
}

pub trait OmegaMatrixFields: sealed::Sealed + Sized {
    fn block_declared(&self) -> &Vec<bool>;
    fn block_declared_mut(&mut self) -> &mut Vec<bool>;
    fn sim_chol(&self) -> &Option<DMatrix<f64>>;
    fn sim_chol_mut(&mut self) -> &mut Option<DMatrix<f64>>;
}
impl OmegaMatrixFields for OmegaMatrix {
    fn block_declared(&self) -> &Vec<bool> {
        &self.block_declared
    }
    fn block_declared_mut(&mut self) -> &mut Vec<bool> {
        &mut self.block_declared
    }
    fn sim_chol(&self) -> &Option<DMatrix<f64>> {
        &self.sim_chol
    }
    fn sim_chol_mut(&mut self) -> &mut Option<DMatrix<f64>> {
        &mut self.sim_chol
    }
}

pub trait RuvMagnitudeFields: sealed::Sealed + Sized {
    fn uses_tad(&self) -> bool;
    fn uses_tad_mut(&mut self) -> &mut bool;
}
impl RuvMagnitudeFields for RuvMagnitude {
    fn uses_tad(&self) -> bool {
        self.uses_tad
    }
    fn uses_tad_mut(&mut self) -> &mut bool {
        &mut self.uses_tad
    }
}

pub trait ThetaBlocksFields: sealed::Sealed + Sized {
    fn decls(&self) -> &Vec<VectorThetaDecl>;
    fn decls_mut(&mut self) -> &mut Vec<VectorThetaDecl>;
    fn level_blocks(&self) -> &Vec<LevelBlockDecl>;
    fn level_blocks_mut(&mut self) -> &mut Vec<LevelBlockDecl>;
}
impl ThetaBlocksFields for ThetaBlocks {
    fn decls(&self) -> &Vec<VectorThetaDecl> {
        &self.decls
    }
    fn decls_mut(&mut self) -> &mut Vec<VectorThetaDecl> {
        &mut self.decls
    }
    fn level_blocks(&self) -> &Vec<LevelBlockDecl> {
        &self.level_blocks
    }
    fn level_blocks_mut(&mut self) -> &mut Vec<LevelBlockDecl> {
        &mut self.level_blocks
    }
}
