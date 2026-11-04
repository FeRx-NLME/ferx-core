# fitrx consolidation, Phase 0: findings

Fact-finding for epic #2693, issue #2694. Nothing in the engine changes.

Measured against ferx-core `eecc0fb7`, ferx-r `main` at `65284c0e` (its lock pins ferx-core `86a05bd5`, built with `nn` and `survival`), and ferxgui `ab9014ca`. R 4.4.2 on macOS arm64, with ferx-r built on the nightly toolchain because the local stable is too old for its lock. Timings are from that machine; sizes should carry to Linux, but no Linux run was made. Scripts are in `plans/assets/fitrx-consolidation-phase0/`.

## Summary

1. **Carrying the bytes is reasonable. Phase 1 does not need to wait on this.** A bundle costs about 54 bytes per observation and 3.4 µs per observation to write. `.rds` files double, because zip bytes do not compress further. `ferx_bootstrap()` is unaffected, because its replicates never become R fits.
2. **The epic's drift list needs three corrections.**
   - `neural_networks` and `method_wall_times_secs` are not "written by R, not read back". R never writes either one.
   - DCM weights are not at risk. They live in `theta`, which round-trips exactly. What R loses is the network metadata, and it loses it on the first save, not only on a re-save.
   - R omits 17 `FitWire` keys, not 9.
3. **"The same converter for fresh and loaded fits" has a gap no phase covers.** `fit_result_to_list` needs a `Population`, and a bundle without `data.csv` has none. See [Gaps for later phases](#gaps-for-later-phases).
4. **There is a new bug.** An R load and re-save of a binary-only fit writes a bundle core refuses (`invalid type: sequence, expected f64`). The cause is `shrinkage_eps`: `NaN` is written as `null`, read back as `numeric(0)`, then written as `[]`. Phase 4 removes the code, so it may not need its own fix.
5. **There are more readers than the epic counts.** ferxgui has its own Rust reader, and it reads `r_extras` directly. It does not check `format_version`, so a v2 bundle that moves those keys would silently blank its verdicts rather than fail. It also assigns into loaded fits, which Phase 5 forbids.

## 1. Field matrix

Core W and Core R mean `save_fit` and `load_fit`. R W and R R mean `ferx_save_fit` and `ferx_load_fit`. The R list column is the key in `fit_result_to_list`. Key sets were extracted from source, then checked against real bundles from eight fixtures: plain, SIR, IOV, SAEM, DCM, nested levels, joint PK-TTE and binary.

### Epic drift list, checked

| Epic claim | Result |
|---|---|
| Never written by R: `environment`, `imp_seed`, `saem_seed`, `npde_seed`, `multi_start_seed`, `n_starts`, `outer_maxiter`, `obs_time_range`, `total_ebe_nm_polished` | Confirmed, and the list is short. Also never written: `method_wall_times_secs`, `neural_networks`, `model_text`, `optimizer`, `outer_gtol`, `inits_from_nca`, `saem_n_subjects_hmc`, `saem_mh_accept_tail`. `model_text` does reach the bundle, as `model.ferx`. |
| Written by R, not read back: `neural_networks`, `method_wall_times_secs` | Wrong. R writes neither. The only key R writes and does not read is `error_model`. `method_wall_times_secs` sits in `.FITRX_ARRAY_KEYS` but nothing emits it. |
| An R load and re-save may lose DCM weights | Wrong on both counts. On `warfarin_dcm`, all 146 `theta` values are `identical()` after an R load. `src/nn/export.rs` does not read `neural_networks`. The metadata (`shape`, activations, `input_center`, `input_scale`, ...) is gone after the first `ferx_save_fit()`, and from core bundles too, since R never reads the key. The visible effect is that `print()` loses its network section and `fit$neural_networks` is `NULL`. |
| Not handled by core: `trace.csv`, `covtab.csv`, `predictions_discrete.csv`, `covariate_types`, `r_extras` | Confirmed. None of the names appears in `src/io/fitrx.rs`, and the binary fixture's core re-save drops both of its extra entries. The list also misses `bayes` and `exclusions`, which R writes and core ignores. |
| Safer in R: data hash check, temp file first | Confirmed. The same gap exists for the model: R checks `model_text` against `model_hash`, and core does not. That answers the open question in #2696. |

One R load and re-save is otherwise stable. Across all eight fixtures the non-JSON entries are byte-identical, and in `fit.json` only `model_path` and `data_path` change, except for `shrinkage_eps` on the binary fixture. Those two paths are rewritten to session temp files, so a re-saved bundle loses its original provenance.

### `fit.json` top-level keys

| `fit.json` key | Core W | Core R | R W | R R | R list key | Note |
|---|---|---|---|---|---|---|
| `method` | yes | yes | yes | yes | `method` |  |
| `method_chain` | yes | yes | yes | yes | `method_chain` |  |
| `converged` | yes | yes | yes | yes | `converged` |  |
| `ofv` | yes | yes | yes | yes | `ofv` |  |
| `ofv_data` | yes | yes | yes | yes | `ofv_data` | core writes only for a priored fit |
| `ofv_prior` | yes | yes | yes | yes | `ofv_prior` | core writes only for a priored fit |
| `prior_summary` | yes | yes | yes | yes | `prior_summary` | core writes only for a priored fit |
| `aic` | yes | yes | yes | yes | `aic` |  |
| `bic` | yes | yes | yes | yes | `bic` |  |
| `n_obs` | yes | yes | yes | yes | `n_obs` |  |
| `n_subjects` | yes | yes | yes | yes | `n_subjects` |  |
| `n_parameters` | yes | yes | yes | yes | `n_parameters` |  |
| `n_iterations` | yes | yes | yes | yes | `n_iterations` |  |
| `interaction` | yes | yes | yes | yes | `interaction` |  |
| `wall_time_secs` | yes | yes | yes | yes | `wall_time_secs` |  |
| `method_wall_times_secs` | yes | yes | **no** | **no** | **none** |  |
| `covariance_wall_time_secs` | yes | yes | yes | yes | `covariance_wall_time_secs` |  |
| `n_threads_used` | yes | yes | yes | yes | `n_threads_used` |  |
| `uses_ode_solver` | yes | yes | yes | yes | `uses_ode_solver` |  |
| `ode_linear_exact` | yes | yes | yes | yes | `ode_linear_exact` |  |
| `uses_sde` | yes | yes | yes | yes | `uses_sde` |  |
| `gradient_method_inner` | yes | yes | yes | yes | `gradient_method_inner` |  |
| `gradient_method_outer` | yes | yes | yes | yes | `gradient_method_outer` |  |
| `nlopt_missing_algorithms` | yes | yes | yes | yes | `nlopt_missing_algorithms` |  |
| `covariance_status` | yes | yes | yes | yes | `covariance_status` |  |
| `covariance_method` | yes | yes | yes | yes | `covariance_method` |  |
| `covariance_n_evals_estimated` | yes | yes | yes | yes | `covariance_n_evals_estimated` |  |
| `trace_path` | yes | yes | yes | yes | `trace_path` | dangling temp path after the run |
| `ebe_convergence_warnings` | yes | yes | yes | yes | `ebe_convergence_warnings` |  |
| `max_unconverged_subjects` | yes | yes | yes | yes | `max_unconverged_subjects` |  |
| `total_ebe_fallbacks` | yes | yes | yes | yes | `total_ebe_fallbacks` |  |
| `total_ebe_nm_polished` | yes | yes | **no** | **no** | **none** |  |
| `warnings` | yes | yes | yes | yes | `warnings` |  |
| `warning_details` | yes | yes | yes | yes | `warning_details` | R writes only when a warning has details |
| `saem_mu_ref_m_step_evals_saved` | yes | yes | yes | yes | `saem_mu_ref_m_step_evals_saved` |  |
| `saem_n_subjects_hmc` | yes | yes | **no** | **no** | `saem_n_subjects_hmc` |  |
| `saem_mh_accept_tail` | yes | yes | **no** | **no** | **none** |  |
| `theta` | yes | yes | yes | yes | see nested |  |
| `omega` | yes | yes | yes | yes | see nested |  |
| `sigma` | yes | yes | yes | yes | see nested |  |
| `error_model` | yes | yes | yes | **no** | **none** | R derives it from `sigma_types`; R never reads it |
| `shrinkage_eps` | yes | yes | yes | yes | `shrinkage_eps` |  |
| `iwres_lag1_r` | yes | yes | yes | yes | `iwres_lag1_r` |  |
| `dw_statistic` | yes | yes | yes | yes | `dw_statistic` |  |
| `covariance_matrix` | yes | yes | yes | yes | `cov_matrix (+`cov_matrix_dim`)` |  |
| `cov_eigenvalues` | yes | yes | yes | yes | `cov_eigenvalues` |  |
| `cov_condition_number` | yes | yes | yes | yes | `cov_condition_number` |  |
| `bic_inputs` | yes | yes | yes | yes | `bic_inputs` |  |
| `profiled_blocks` | yes | yes | yes | yes | `profiled_blocks` |  |
| `profiled_levels_at_bound` | yes | yes | yes | yes | `profiled_levels_at_bound` |  |
| `sir` | yes | yes | yes | yes | see nested |  |
| `iov` | yes | yes | yes | yes | see nested |  |
| `eta_param_info` | yes | yes | yes | yes | `eta_param_types, eta_linked_theta` | R writes `individual_param_name` as `""` |
| `model_name` | yes | yes | yes | yes | `model_name` |  |
| `ferx_version` | yes | yes | yes | yes | `ferx_version` |  |
| `environment` | yes | yes | **no** | **no** | **none** | includes `username` |
| `model_path` | yes | yes | yes | yes | `model_path` |  |
| `data_path` | yes | yes | yes | yes | `data_path` |  |
| `model_hash` | yes | yes | yes | yes | `model_hash` |  |
| `data_hash` | yes | yes | yes | yes | `data_hash` |  |
| `model_text` | yes | yes | **no** | **no** | `model_text` | R writes the same text as `model.ferx` only; core overwrites from `model.ferx` on load |
| `theta_init` | yes | yes | yes | yes | `theta_init` |  |
| `omega_init` | yes | yes | yes | yes | `omega_init (+`omega_init_dim`)` |  |
| `left_init` | yes | yes | yes | yes | `left_init` |  |
| `stationary_start` | yes | yes | yes | yes | `stationary_start` |  |
| `omega_is_diagonal` | yes | yes | yes | yes | `omega_is_diagonal` |  |
| `kappa_is_diagonal` | yes | yes | yes | yes | `kappa_is_diagonal` |  |
| `sigma_init` | yes | yes | yes | yes | `sigma_init` |  |
| `obs_time_range` | yes | yes | **no** | **no** | `obs_time_range` |  |
| `final_gradient` | yes | yes | yes | yes | `final_gradient` |  |
| `final_gradient_source` | yes | yes | yes | yes | `final_gradient_source` |  |
| `optimizer` | yes | yes | **no** | **no** | `optimizer_label` |  |
| `n_starts` | yes | yes | **no** | **no** | `n_starts` |  |
| `multi_start_seed` | yes | yes | **no** | **no** | `multi_start_seed` |  |
| `saem_seed` | yes | yes | **no** | **no** | `saem_seed` |  |
| `sir_seed` | yes | yes | yes | yes | `sir_seed_used` | R writes only beside `sir.settings` |
| `scoring_settings` | yes | yes | yes | yes | `scoring_settings` |  |
| `imp_seed` | yes | yes | **no** | **no** | `imp_seed` |  |
| `npde_seed` | yes | yes | **no** | **no** | **none** |  |
| `bloq_method` | yes | yes | yes | yes | `bloq_method_label` | R omits when empty |
| `outer_maxiter` | yes | yes | **no** | **no** | `outer_maxiter` |  |
| `outer_gtol` | yes | yes | **no** | **no** | `outer_gtol` |  |
| `inits_from_nca` | yes | yes | **no** | **no** | `inits_from_nca` |  |
| `covariate_names` | yes | yes | yes | yes | `covariate_names` |  |
| `input_columns` | yes | yes | yes | yes | `input_columns` |  |
| `neural_networks` | yes | yes | **no** | **no** | `neural_networks` | **R drops it on every save**, live fit included |
| `data_bindings` | yes | yes | yes | yes | `theta_levels, covariate_stats` | R rebuilds it from `theta_levels` |
| `reader_settings` | yes | yes | yes | yes | `reader_settings` | R carries it as a JSON string |
| `population_fingerprint` | yes | yes | yes | yes | `population_fingerprint` | R carries it as a JSON string; O(N) in size |
| `iov_occasion` | yes | yes | yes | yes | `iov_occasion` |  |
| `packed_estimate` | yes | yes | yes | yes | `packed_estimate` |  |
| `mixture_overrides` | yes | yes | yes | yes | `mixture_overrides` |  |
| `vi` | yes | yes | yes | yes | `vi` |  |

R also writes three top-level keys core does not have: `bayes`, `exclusions` and `covariate_types`. It reads all three back. Core ignores them on load. `r_extras` is covered below.

### Nested keys

Core writes and reads every key below. Rows list only what differs or needs a name mapping.

| Block | R W and R R | R list keys |
|---|---|---|
| `theta.{names,estimates,se,fixed,transform}` | yes | `theta_names`, `theta`, `se_theta`, `theta_fixed`, `theta_transforms` |
| `omega.{names,matrix,se,fixed,log_transformed,param_corr,shrinkage,init_as_sd}` | yes | `eta_names`, `omega`, `se_omega`, `omega_fixed`, `eta_log_transformed`, `omega_param_corr`, `shrinkage_eta`, `omega_init_as_sd` |
| `sigma.{names,estimates,se,fixed,types,init_as_sd}` | yes | `sigma_names`, `sigma`, `se_sigma`, `sigma_fixed`, `sigma_types`, `sigma_init_as_sd` |
| `sigma.{residual_correlations,residual_correlation_fixed,se_residual_correlations}` | yes; R rebuilds 0-based indices | `residual_correlation_{i,j,rho,fixed,names}`, `se_residual_correlation` |
| `sir.{ci_theta,ci_omega,ci_sigma,ci_kappa,ess}` | yes | `sir_ci_*`, `sir_ess` |
| `sir.resamples_packed` | yes | `sir_resamples`, `sir_resamples_n`, `sir_resamples_dim` |
| `sir.settings`, `scoring_settings` | yes, through the glue's `SIR_KEYS` and `SCORING_KEYS` tables | `sir_settings`, `scoring_settings` |
| `iov.*` | yes, all keys; `levels` becomes `ebe_kappas_by_level` | same names, plus `omega_iov_dim` |
| `vi.*`, `vi.eta_posterior[].*` | yes, all keys | `vi`, reshaped by `.ferx_vi_from_raw()` |
| `eta_param_info[].{eta_name,param_type,linked_theta}` | yes | `eta_param_types`, `eta_linked_theta` |
| `eta_param_info[].individual_param_name` | **R writes `""` and never reads it** | none |
| `data_bindings.levels` | yes | `theta_levels`, after laying out on the model |
| `data_bindings.covariate_stats` | yes | `covariate_stats` |
| `environment.{os,arch,in_docker,username}` | **no** | none |

`environment.username` deserves a decision. Once bytes ride on the R fit, the username goes into every `.rds` a user shares. Absolute model and data paths are already there.

### Bundle entries

| Entry | Core W | Core R | R W | R R | R list key |
|---|---|---|---|---|---|
| `manifest.json` | yes | yes | yes | yes | none |
| `fit.json` | yes | yes | yes | yes | see above |
| `ebes.csv` | yes | yes | yes, plus `n_obs_discrete` | yes | `ebe_etas`, `subject_ids`, `subject_summary` |
| `ebes_kappa.csv` | yes | yes | yes | yes | `ebe_kappas` |
| `ebes_kappa_<LEVEL>.csv` | yes | yes | yes | yes | `ebe_kappas_by_level` |
| `conddist.csv` | yes | yes | yes | yes | `cond_dist`; `nsamp` and `burnin` are lost by both |
| `predictions.csv` | yes | yes | yes | yes | `sdtab`; no TTE rows from either writer |
| `predictions_discrete.csv` | **no** | **no** | yes | yes | `sdtab` |
| `covtab.csv` | **no** | **no** | yes | yes | `covtab` |
| `trace.csv` | **no** | **no** | yes | yes | `trace` |
| `model.ferx` | yes | yes | yes | yes | `model_text`, `model_source` |
| `warnings.txt` | yes | no, a mirror | yes | no, a mirror | none |
| `data.csv` | optional | yes | optional | yes | `data_path`, `data_source` |

### `r_extras`

Core neither writes nor reads any of these. ferxgui reads the three marked.

| Key | R W | R R | R list | What it is | Proposed Phase 1 home |
|---|---|---|---|---|---|
| `stalled_at_init` (ferxgui) | yes | yes | yes | `model_selection::stalled_at_init(&FitResult)` | schema slot |
| `estimate_near_boundary` (ferxgui) | yes | yes | yes | `model_selection::estimate_near_boundary(&FitResult)` | schema slot |
| `max_abs_correlation` (ferxgui) | yes | yes | yes | `model_selection::max_abs_correlation(&FitResult)` | schema slot |
| `cov_fixed` | yes | yes | yes | glue's `cov_matrix_held(model, ...)` | schema slot |
| `model_structure` | yes | yes | yes | derived from the model by the glue | recompute from `model.ferx`, or `interface_extras` |
| `theta_levels` | yes | checked only | yes | copy of `data_bindings.levels`, kept "for one release" | delete |
| `call_settings` | yes | yes | added in R | R arguments | `interface_extras` |
| `model_file_settings` | yes | yes | added in R | R-only | `interface_extras` |
| `data_name` | yes | yes | added in R | R-only | `interface_extras` |
| `gradient`, `gradient_used` | yes | yes | added in R | R labels | `interface_extras` |
| `model_file_path` | listed | yes | no | nothing ever sets it | delete |

The three `model_selection` functions take only a `FitResult`. A comment in `.fitrx_collect_r_extras()` says two of them cannot be recomputed from the wire, but it predates `warning_details` (#2268). Phase 1 can derive them at write time.

### In the R list with no home in any bundle

| Field | Source | Note |
|---|---|---|
| `importance_sampling`, `impmap_trace` | engine | Lost by both writers. |
| `individual_estimates` | glue | Absent after an R load. |
| `warnings_severity`, `_category`, `_message`, `_source_method`, `warnings_structured` | glue and R | Rebuilt from `warnings` and `warning_details`. |
| `eta_normality`, `cor_matrix`, `estimates`, `eta_cov` | R | Derived. All but `eta_normality` are recomputed on load. |
| `cov_labels` | glue | Needs the model. |

`FitResult` itself has no wire slot for `bayes`, `exclusions`, `importance_sampling`, `impmap_trace`, `covariate_table` or `covariate_relations`. If the bytes are the source of truth, Phase 1 has to add the first five, or the view loses them on a fresh fit too. #2695 lists neither `bayes` nor `exclusions`.

## 2. R mutator review

The scan walks the parsed AST of all 73 files in `R/` (`mutators.R`) instead of grepping, so it sees every variable name and `$`, `[[`, `[`, `attr<-`, `names<-` and `class<-` targets. It found 774 compound assignments. Those whose target is a fit, under any name:

| Function | Sites | When it runs | Verdict |
|---|---|---|---|
| `.ferx_fit_from_raw` (`internal-fit-format.R`) | 100 | before the class is set (`:1184`) | Stays. This is the view builder. |
| `.ferx_apply_cov_sentinels` (`ferx_fit.R`) | 5 | before the class from `ferx_fit`; **after** it from `ferx_covariance` | Stays as view code; the second caller changes with `ferx_covariance`. |
| `.ferx_populate_derived_fields` (`internal-persist.R`) | 3 | same | Same. |
| `ferx_load_fit` | 28 | before the class (`:281`) | Removed in Phase 3. |
| `.fitrx_wire_to_fit` (variable `out`) | 52 | before the class | Removed in Phase 3. |
| `.fitrx_vi_from_wire` (variable `raw`) | 8 | before the class | Removed in Phase 3. |
| `ferx_covariance` | 23 | after the class | **Has to change.** |
| `ferx_sir` | 18 | after the class | **Has to change.** |
| `ferx_calc_npde` (`fit$sdtab`) | 1 | after the class | **Has to change.** |
| `ferx_collect` (`data_path`, `trace_path`, `trace`) | 3 | after the class | **Has to change.** |

That is 108 sites before the class, 88 removed, and 45 to change. The issue's first counts were close for the four named files: 100 against 81, 28 against 25, 23 against 22, and 18 against 18. Widening the search added `out` and `raw` in the loader but found no further post-class writers. Other hits are not fits: `res` in `ferx_bootstrap`, `result` in the search tools (a container holding a fit), and `ebes` in the writer.

What the four changes need:

- **`ferx_covariance` and `ferx_sir`** rebuild a skeleton `FitResult` from about 35 R arguments (`default_fit_result`, `FitSkeletonInputs`), run the step, then merge the result into the R list by hand. Under decision 1 they should load the fit's bytes, run the core step, and return a new fit built by the converter. That deletes the skeleton and the merge, including the warning filtering. It belongs in Phase 5 at the latest, and it is the largest item in the epic.
- **`ferx_calc_npde`** adds `NPDE` and `NPD` to `sdtab`. Core's `predictions.csv` already has those columns, so the same treatment works. Otherwise NPDE stays an added object.
- **`ferx_collect`** rewrites `data_path` after a background job, because the worker's temp directory is gone. Either it becomes a view-only field or the fit is rebuilt. `trace` and `trace_path` are stripped when the caller did not ask for a trace, so the job should not ask core to keep them.

Outside `R/`:

- **ferx-r tests:** about 353 assignments into fit-like objects across 48 files, mostly to build edge cases. The heaviest files are `test-ferx_runlog.R` (55), `test-posthoc-settings.R` (33) and `test-ferx_xpose.R` (25). This count is heuristic, by key name. Phase 5 needs a test-only escape hatch or a large rewrite, which should be sized before it starts.
- **Vignettes and README:** none.
- **ferxgui:** seven sites, covered below.

## 3. Cost of carrying bundle bytes

Datasets are `warfarin` with per-subject noise on DV, time and dose, 11 observations per subject, run at `maxiter = 0`. My first attempt copied subjects verbatim, and deflate shrank that to a fifteenth of the real size, so do not reuse that approach. Core figures come from a small harness calling `save_fit` and `load_fit` directly, as the median of five runs (`harness/`). No in-memory save exists yet, so they include writing to disk. Bundles exclude `data.csv` unless noted.

### Bundle size and time

| Subjects | Observations | Bundle | With `data.csv` | Core save | Core load | R save today | R load today |
|---|---|---|---|---|---|---|---|
| 10 | 110 | 10.3 kB | 11.4 kB | 0.7 ms | 0.3 ms | 13 ms | 14 ms |
| 1,000 | 11,000 | 597 kB | 692 kB | 36 ms | 15 ms | 294 ms | 321 ms |
| 10,000 | 110,000 | 5.9 MB | 6.9 MB | 371 ms | 155 ms | 3.5 s | 3.9 s |
| 50,000 | 550,000 | 29.6 MB | 34.3 MB | 1.87 s | 0.78 s | 18.6 s | 24.5 s |

Size and time are linear in observations. At 10,000 subjects, `predictions.csv` is 86% of the compressed bundle. `population_fingerprint` is 2.8 MB of the 3.4 MB uncompressed `fit.json`.

Core's writer is 8 to 19 times faster than R's. The load figures are not like for like, because R's includes building the list. Even so, Phase 4 should make saving much faster.

### Effect on the R object and `.rds`

| Subjects | `object.size` | With bytes | `.rds` | With bytes | `saveRDS` | With bytes | `readRDS` | With bytes |
|---|---|---|---|---|---|---|---|---|
| 10 | 69 kB | 79 kB (+15%) | 10.9 kB | 21.2 kB | 1 ms | 1 ms | under 1 ms | 1 ms |
| 1,000 | 1.6 MB | 2.2 MB (+37%) | 603 kB | 1.2 MB | 35 ms | 45 ms | 3 ms | 4 ms |
| 10,000 | 15.7 MB | 21.6 MB (+38%) | 5.9 MB | 11.8 MB | 374 ms | 471 ms | 31 ms | 34 ms |
| 50,000 | 78 MB | 108 MB (+38%) | 29.2 MB | 58.7 MB | 1.83 s | 2.34 s | 161 ms | 174 ms |

`.rds` size doubles at every scale, because zip bytes do not compress further. That is the main visible cost: at 10,000 subjects it is 11.8 MB, against a 3.5 MB data file. Users who want the small form have `.fitrx`.

Retained SIR draws add about 9 bytes per cell: 5,000 draws of 7 parameters took a warfarin bundle from 10 kB to 324 kB. Extrapolating, a 10,000 × 50 pool (#556) would be about 4.5 MB.

### Running many fits

| Path | Effect |
|---|---|
| `ferx_bootstrap()` | None. Replicates run inside ferx-tools, and the binding returns tables. `fit_result_to_list` has two callers: `ferx_rust_fit` and the search tools' final fit. |
| `ferx_covsearch()` and the other search tools | One or two bundles per search, for the final fits only. Candidates never reach R. |
| `ferx_fit()` in a loop, warfarin, 10 subjects | 24 ms per fit, so +0.7 ms, about 3%. 50 retained fits grow from 3.3 MB to 3.8 MB. |
| `ferx_fit()` on 1,000 subjects, FOCE, 38 iterations, converged | 0.36 s per fit, so +36 ms, about 10%. This is the worst realistic ratio I found, on a fast closed-form model. ODE models will sit far lower. |
| `ferx_fit_async()` | One `saveRDS` per job, so the table above applies once. |

Not measured: the copy of `Vec<u8>` into an R raw vector across the FFI. It is one `memcpy`, and I expect it to be negligible beside the deflate cost.

### Verdict

Carrying bytes is reasonable, and I am not raising a blocker. Two things to decide before Phase 1:

- **Deflate on every fit is avoidable.** If the 10% case matters, the bundle could use a faster compression level in memory and recompress at save. I would not do this until someone complains.
- **Two identical bundles will differ.** `manifest.json` has `created_at`, so tamper detection for decision 2 cannot compare a re-serialization to the stored bytes. It has to compare the view with what the bytes decode to, or hash the view at build time.

## 4. Consumers of `.fitrx`

### ferxgui

The repository is `Robterheine/ferxgui`, not under FeRx-NLME, and it is maintained outside this project. It uses the format in two ways.

**Its own Rust reader, `src/io/fitrx.rs` and `src/io/bundle.rs`.** This is a third implementation with its own wire structs. It reads `fit.json`, `ebes.csv`, `predictions.csv`, `covtab.csv`, `conddist.csv`, `trace.csv` and `warnings.txt`, and it shows `manifest.json` in a bundle inspector.

- It never checks `format_version`, only displays it, and every field is `#[serde(default)]`. A v2 bundle will parse, and anything renamed or moved reads as absent with no error.
- It reads `r_extras.estimate_near_boundary`, `stalled_at_init` and `max_abs_correlation`. Moving those to schema slots in Phase 1 blanks its boundary and stall verdicts silently.
- Its wire expects `warnings_structured`, a key neither writer emits today.
- It tolerates scalars where arrays belong, from before #379, so stricter arrays in v2 are safe for it.
- `tests/fixtures/` holds eight v1 bundles, and `make_fixtures.R` rebuilds them through R.

**Through R, `src/io/r_extract.rs`.** It runs `ferx_fit()` and `ferx_save_fit()` for every run, and calls `ferx_load_fit()` in 15 places. Phase 5 breaks seven assignments:

- `fit$data_path <- data_override` at `:177`, `:339` and `:850`, and the restore at `:347`. This relocates a moved project, and it is a genuine use case. `ferx_covariance()` has no `data` argument, so Phase 5 needs one, or a supported way to rebind the path.
- `fit$sir_resamples`, `_n` and `_dim` at `:769`-`:771`. These re-inject SIR draws on the stated grounds that `ferx_load_fit()` does not restore them. That has been untrue since ferx-r #549, so the code can simply go.

The "recompute standard errors" flow is load, relocate, `ferx_covariance()`, save, reload to verify, then swap with a `.bak`. It depends on a post-hoc step producing a saveable fit, so that has to keep working.

Decision 3 also splits the GUI: its Rust reader will keep opening v1 bundles while its R calls refuse them. The maintainer needs notice before the ferx-r lock moves.

### Other consumers

| Consumer | Use | Impact |
|---|---|---|
| `ferx-cli` | `--output` save; bundle summary load (`main.rs:524`); `export-dcm`; `gam --from-fit` | Goes through core. Old bundles start failing. |
| `[priors] from_fit` (`src/io/fit_estimates.rs`) | Reads a `.fitrx` during model parsing | **Not in the epic.** A model that names an old bundle stops parsing after the bump, with a version error. The book documents this feature (chapter 25). |
| `ferx-r` | `ferx_fit(output =)`, `ferx_save_fit`, `ferx_load_fit` | The epic's subject. |
| `InsightRX/luna` (public) | `ferx_load_fit()` in `get_status`, `runs_as_table` and `luna_ferx_info` | Goes through R. It loads the full fit just to check status, so Phase 3's speed-up helps it. Old runs in a project stop loading. |
| Private InsightRX repositories | One parses bundles directly, one writes them through `ferx_fit(output =)` | Not detailed in this public report. |
| `ferx-book`, `ferx-nlme.github.io` | Prose and examples | No format dependency. `ferx-book` chapter 13 describes the zip layout. |
| `ferx-tools` search journal | `FitResult` JSON, not `.fitrx` | Unaffected. It is a separate persistence format the epic leaves alone. |

## Gaps for later phases

These are not in the phase issues as written.

1. **The converter needs a `Population`.** `fit_result_to_list` takes `(result, population, model, bindings)`. `sdtab()` reads `TIME`, `DV`, `OCC` and `CMT` from the population, and `ebe_kappas` reads occasion labels from it. #2696's note that only `write_predictions_csv` needs a `Population` is true of `save_fit`, not of this converter. A bundle saved without data has no population, so Phase 3 cannot use the same converter unless Phase 1 stores those columns or the converter is split to read them from `predictions.csv`.
2. **The converter needs the model too.** `cov_fixed`, `cov_labels`, `model_structure` and `theta_levels` come from a parsed model. Today a load survives a model the engine can no longer parse (`unavailable_theta_levels`, #531). Phase 3 has to keep that path, and a `[covariate_nn]` model does not parse in a build without `nn`.
3. **Discrete and TTE rows.** Core's `load_fit` sets `discrete_rows` and `tte_rows` to empty, and neither writer keeps TTE rows. #2695 covers `predictions_discrete.csv` but not TTE, so `sdtab` on a loaded joint fit will still differ (ferx-r#689).
4. **R-only fields arrive after the bytes exist.** `call_settings`, `data_name`, `gradient` and the like are added by `.ferx_fit_from_raw` after the glue returns, and R appends the high-condition-number warning in `.ferx_apply_cov_sentinels`. So the bytes made in the glue will not hold them. #2696's "update `interface_extras` on existing bytes" covers the first group. The warning should move into core.
5. **`[priors] from_fit`, `bayes`, `exclusions`** as noted above.
6. **ferxgui** needs a heads-up, and Phase 5 needs a supported way to relocate a data path.
