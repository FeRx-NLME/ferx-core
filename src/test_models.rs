//! Model texts shared by test modules: each was written out byte-identically in two or
//! more tests, so it lives here once. A test that needs a variant writes its own text.

/// Used by src/api/tests/adaptive_assay_magnitude_tests.rs, src/api/tests/adaptive_sim_tests.rs.
pub(crate) const ODE_1: &str = r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
";

/// Used by src/api/tests/adaptive_cmt_toggle_tests.rs.
pub(crate) const ODE_2: &str = r"
[parameters]
  theta TVKA(0.3, 0.01, 10.0)
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  KA = TVKA
  CL = TVCL * CRCL / 100.0
  V  = TVV
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL / V) * central / (1 + central / 200)
[scaling]
  y[CMT=1] = depot
  y[CMT=2] = central
[error_model]
  DV ~ proportional(PROP)
";

/// Used by src/api/tests/iov_integration.rs.
pub(crate) const ODE_3: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  block_sigma (PROP_ERR_UNBOUND, PROP_ERR_TOTAL) = [
    0.04,
    0.01, 0.09
  ]

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  y[CMT=1] = 2.0 * central / V
  y[CMT=2] = central / V

[error_model]
  CMT=1: DV ~ proportional(PROP_ERR_TOTAL)
  CMT=2: DV ~ proportional(PROP_ERR_UNBOUND)
";

/// Used by src/api/tests/iov_integration.rs.
pub(crate) const ONE_CPT_IV_1: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 10.0) FIX
  theta TVV(10.0, 0.1, 100.0) FIX
  omega ETA_CL ~ 0.04 FIX
  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.00] FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
";

/// Used by src/api/tests/adaptive_sim_tests.rs, src/api/tests/non_fit_diagnostics_tests.rs.
pub(crate) const ODE_4: &str = r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_method = rk45
  ode_max_steps = 2
";

/// Used by src/api/tests/adaptive_moved_reset_tests.rs, src/api/tests/adaptive_sim_tests.rs, src/api/tests/non_fit_diagnostics_tests.rs.
pub(crate) const ODE_5: &str = r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
";

/// Used by src/api/tests/nonfinite_objective_tests.rs.
pub(crate) const ONE_CPT_IV_2: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  omega ETA_CL ~ 0.04
  sigma ADD ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ additive(ADD)
";

/// Used by src/api/tests/nonfinite_objective_tests.rs, src/api/tests/ode_solver_diagnostics_tests.rs, src/api/tests/pool_plan_tests.rs, src/api/tests/quiet_fit_tests.rs, src/api/tests/stalled_at_init_warning_tests.rs.
pub(crate) const ONE_CPT_IV_3: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP)
";

/// Used by src/api/tests/tests_derived_iov_kappa.rs.
pub(crate) const ONE_CPT_IV_4: &str = r"
[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[initial_conditions]
  init(central) = 30 * V

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Used by src/estimation/bayes.rs, src/estimation/impmap.rs, src/estimation/importance_sampling.rs.
pub(crate) const ONE_CPT_IV_5: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

/// Used by src/estimation/bayes.rs, src/estimation/impmap.rs.
pub(crate) const ONE_CPT_IV_6: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.04

[mixture]
  nsub = 2
  logit(1) = MIXL
  omega(2) ETA_CL ~ 0.15

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

/// Used by src/estimation/impmap.rs, src/estimation/saem.rs.
pub(crate) const ONE_CPT_IV_7: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta LOGIT_F(-0.405465, -10.0, 10.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_F ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 else TVCL2
  V  = TVV
  F  = inv_logit(LOGIT_F + ETA_F)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V, f=F)

[error_model]
  DV ~ proportional(EPS)
";

/// Used by src/estimation/impmap.rs, src/estimation/saem.rs.
pub(crate) const ONE_CPT_IV_8: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta LOGIT_F(0.5, 0.0, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_F ~ 0.04
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  F  = inv_logit(LOGIT_F + ETA_F)
  CL = TVCL * exp(ETA_CL) * F
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

/// Used by src/estimation/impmap.rs, src/estimation/saem.rs.
pub(crate) const ONE_CPT_IV_9: &str = r"
[parameters]
  theta TVCL(4.0, 0.0, 100.0)
  theta TH_CRCL(0.02, 0.0, 5.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.01 FIX

[individual_parameters]
  CL = (TVCL + (CRCL - 90.0) * TH_CRCL) * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

/// Used by src/estimation/impmap.rs, src/estimation/saem.rs.
pub(crate) const ONE_CPT_IV_10: &str = r"
[parameters]
  theta TVCL(4.0, 0.0, 100.0)
  theta TH_CRCL(0.2, 0.0, 5.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.01 FIX

[individual_parameters]
  CL = (TVCL + (CRCL - 90.0) * TH_CRCL) * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

/// Used by src/estimation/inner_optimizer_tests.rs, src/markov/endpoint.rs.
#[cfg(feature = "markov")]
pub(crate) const MODEL_1: &str = r"
[parameters]
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
  omega ETA_Q ~ 0.1

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = exp(LQ01 + ETA_Q)
  transition asleep -> awake  = exp(LQ10)
";

/// Used by src/estimation/bayes.rs, src/estimation/mixture.rs.
pub(crate) const ONE_CPT_IV_11: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta P1(0.5, 0.001, 0.999)
  omega ETA_CL ~ 0.05
  kappa KAPPA_CL ~ 0.02
  sigma EPS ~ 0.04

[mixture]
  nsub = 2
  p(1) = P1

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL + KAPPA_CL) else TVCL2 * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

/// Used by src/api/tests/tests_mixture_postfit.rs, src/estimation/mixture.rs.
pub(crate) const ONE_CPT_IV_12: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta P1(0.5, 0.001, 0.999)
  omega ETA_CL ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  p(1) = P1

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

/// Used by src/estimation/covariance.rs, src/estimation/mixture.rs.
pub(crate) const ONE_CPT_IV_13: &str = r"
[parameters]
  theta TVCL(1.5, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.2, -10.0, 10.0)
  theta BWT(0.05, -5.0, 5.0)
  omega ETA_CL ~ 0.06
  sigma EPS ~ 0.02

[mixture]
  nsub = 2
  logit(1) = MIXL + BWT*(WT - 75)
  omega(2) ETA_CL ~ 0.15
  sigma(2) EPS ~ 0.03

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

/// Used by src/estimation/fixed_eta_gradient_tests.rs, src/estimation/saem_mstep_sa_tests.rs.
pub(crate) const ONE_CPT_IV_14: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  sigma PROP_ERR ~ 0.10 (sd)
  sigma ADD_ERR  ~ 0.50 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ combined(PROP_ERR * (1.0 + 0.5 * WPSE), ADD_ERR) weight = WPSE
[covariates]
  WPSE continuous
";

/// Used by src/estimation/vi/elbo_tests.rs, src/estimation/vi/run_tests.rs.
pub(crate) const ONE_CPT_IV_15: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Used by src/estimation/vi/elbo_tests.rs, src/estimation/vi/run_tests.rs.
pub(crate) const ONE_CPT_ORAL_1: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  theta TVKA(1.5, 0.01, 50.0)
  block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Used by src/estimation/vi/elbo_tests.rs, src/estimation/vi/run_tests.rs.
pub(crate) const ONE_CPT_IV_16: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  iov_column = OCC
";

/// Used by src/estimation/vi/run_tests.rs.
pub(crate) const ONE_CPT_IV_17: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0) FIX
  theta TVV(10.0, 1.0, 100.0) FIX
  omega ETA_CL ~ 0.09 FIX
  omega ETA_V  ~ 0.04 FIX
  sigma PROP_ERR ~ 0.04 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Used by src/frem/mod.rs.
pub(crate) const ONE_CPT_ORAL_2: &str = r"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Used by src/estimation/mixture.rs, src/parser/kernel_jit_fallback_tests.rs.
pub(crate) const ONE_CPT_IV_18: &str = r"
[parameters]
  theta TVCL1(1.2, 0.01, 100.0)
  theta TVCL2(2.5, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  theta BWT(0.0, -5.0, 5.0)
  omega ETA_CL ~ 0.05
  sigma EPS ~ 0.01

[mixture]
  nsub = 2
  logit(1) = MIXL + BWT*(WT - 75)

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

/// Used by src/api/tests/tests.rs, src/estimation/covariate_mu_ref_tests.rs, src/estimation/impmap.rs, src/parser/model_parser_tests.rs.
pub(crate) const ONE_CPT_IV_19: &str = r"
[parameters]
  theta TVCL(150.0, 0.0, 1000.0)
  theta TH_CRCL(2.0, 0.0, 50.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.2
  omega ETA_V ~ 0.1
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = (TVCL + (CRCL - 90.0) * TH_CRCL) * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

/// Used by src/parser/model_parser_tests.rs.
pub(crate) const ONE_CPT_IV_20: &str = r"
[parameters]
  theta THETA_F(0.70, 0.001, 0.999)
  sigma EPS ~ 0.01
  omega ETA_F  ~ 0.1

[individual_parameters]
  F = inv_logit(logit(THETA_F) + ETA_F)

[structural_model]
  pk one_cpt_iv(cl=1, v=1)

[error_model]
  DV ~ proportional(EPS)
";

/// Used by src/pk/mod.rs.
pub(crate) const ONE_CPT_IV_21: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.1
[individual_parameters]
  CL = TVCL * exp(ETA_CL) * (WT/70)
  V  = TVV * (AGE/40)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ additive(EPS)
";

/// Used by src/sens/ode_adjoint_phenomena_tests.rs.
pub(crate) const ONE_CPT_ORAL_3: &str = r"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVLAG(0.5, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.1
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  KA = TVKA
  ALAG = TVLAG * exp(ETA_LAG)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=ALAG)
[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Used by src/sens/ode_adjoint_phenomena_tests.rs, src/sens/ode_adjoint_tests.rs.
pub(crate) const ODE_6: &str = r"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.06
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Used by src/estimation/agq.rs, src/estimation/covariance_salvage_tests.rs, src/sens/ode_adjoint_phenomena_tests.rs, src/sens/ode_adjoint_tests.rs.
pub(crate) const ONE_CPT_ORAL_4: &str = r"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Used by src/sens/ode_adjoint_phenomena_tests.rs, src/sens/ode_adjoint_tests.rs.
pub(crate) const ODE_7: &str = r"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVLAG(0.4, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.1
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  KA = TVKA
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Used by src/estimation/inner_optimizer_iov_tests.rs, src/sens/ode_adjoint_tests.rs.
pub(crate) const ODE_8: &str = r"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(30.0, 1.0, 500.0)
  theta FU(0.3, 0.01, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.06
  kappa KAPPA_CL ~ 0.04
  block_sigma (PROP_TOT, PROP_UNB) = [0.01, 0.006, 0.0225]
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V = TVV * exp(ETA_V)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y[CMT=1] = central / V
  y[CMT=2] = FU * central / V
[error_model]
  CMT=1: DV ~ proportional(PROP_TOT)
  CMT=2: DV ~ proportional(PROP_UNB)
[fit_options]
  iov_column = OCC
";

/// Used by src/stats/likelihood.rs.
#[cfg(feature = "survival")]
pub(crate) const ODE_9: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta TVH0(0.01, 1e-5, 10.0)
  theta TVBETA(0.5, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central
[event_model]
  cmt    = 2
  hazard = H0 * exp(BETA * (central / V))
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
";

/// Used by src/api/tests/vpc_cif_tests.rs, src/api/tests/vpc_kmmc_tests.rs, src/api/tests/vpc_rtte_tests.rs, src/api/tests/vpc_tte_tests.rs, src/stats/npde_tte_tests.rs.
#[cfg(feature = "survival")]
pub(crate) const MODEL_2: &str = r"
[parameters]
  theta TVLAMBDA_A(0.05, 0.001, 10.0)
  theta TVLAMBDA_B(0.03, 0.001, 10.0)

[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA_A

[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = TVLAMBDA_B
";

/// Used by src/api/tests/vpc_cif_tests.rs, src/api/tests/vpc_rtte_tests.rs, src/api/tests/vpc_tte_tests.rs, src/stats/npde_tte_tests.rs.
#[cfg(feature = "survival")]
pub(crate) const ONE_CPT_IV_22: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
";
