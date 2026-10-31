//! Structured diagnostics for model validation.
//!
//! [`Diagnostic`] is the shared currency between the validation logic in
//! [`crate::api`] / [`crate::parser`] and the `ferx check` CLI command. It
//! replaces (well, wraps) the historical free-text `Result<_, String>` errors
//! with a machine-readable shape — a stable `code`, the owning block, a
//! block-level `line`, and an optional `suggestion` — so external callers
//! (coding agents, language bindings) can act on validation output
//! programmatically instead of regex-matching prose.
//!
//! The same `Diagnostic`s feed both `ferx check` (which collects *all* of them
//! in one pass) and `fit()` (which still hard-errors on the first one, via
//! [`first_error`]), keeping a single source of truth for the check logic.
//!
//! ## Error-code registry
//!
//! Codes are stable identifiers (prefix `E_` for errors, `W_` for warnings and for the
//! few informational rows, which carry [`Severity::Info`] — `W_LEVEL_SUMMARY`, #2387).
//! The table is exhaustive: every code the engine emits — as a [`Diagnostic`]'s or an
//! `EngineError`'s code, or as the code opening a `FitResult.warnings` or data-reader
//! message — has a row here **and** in `docs/file-formats/check-report.qmd`, and
//! neither lists a code nothing emits. `diagnostics_code_registry_tests.rs` enforces
//! both (#1051) by scanning the non-test code under `src/` for code-shaped tokens, so
//! write each code as a literal: one assembled at run time is invisible to it.
//!
//! | code | meaning |
//! |------|---------|
//! | `E_PARSE`                 | the model file failed to parse; on block `data_selection`, a `[data_selection]` clause that does not compile into a row filter, which `run_sir` / `run_covariance` reach from recorded reader settings edited after the fit (#1774) |
//! | `E_MISSING_BLOCK`         | a required `[block]` is absent; on block `simulation`, `run_model_simulate` on a model file with no `[simulation]` block (#1788) |
//! | `E_UNKNOWN_BLOCK`         | a `[block]` header is not a recognised block name |
//! | `E_DEPRECATED_BLOCK`      | a `[block]` that was ferx syntax and is no longer read |
//! | `E_BLOCK_INSTANCE_NAME`   | a `[block NAME]` instance name is present where none is taken, or missing where one is required |
//! | `E_BLOCK_FEATURE_DISABLED`| a `[block]` needs a cargo feature this binary was not built with |
//! | `E_NN_FEATURE_DISABLED`   | a `[covariate_nn]` block needs `--features nn` |
//! | `E_MISSING_COVARIATE`     | the model references a covariate not present in the data; on block `covariates`, the strict reader found a declared covariate column missing, whether `ferx check --data` read the file or `run_sir` / `run_covariance` re-read `fit.data_path` (#1774); on block `simulation`, `run_model_simulate`'s `[simulation]` design does not state a covariate the model references (#1083, #1788) |
//! | `E_COVARIATE_NOT_NUMERIC` | the strict reader found a declared covariate column holding a value that is not a number. Block `covariates`. Raised by `ferx check --data` and by the `run_sir` / `run_covariance` re-read of `fit.data_path` (#1774), through one classifier |
//! | `E_ODE_METHOD_LINEAR`     | `ode_method = linear` on a model that cannot take the exact linear path |
//! | `E_PER_CMT_SCALING`       | an observed compartment lacks a per-CMT scaling entry |
//! | `E_PER_CMT_ERROR_MODEL`   | an observed compartment lacks a per-CMT `[error_model]` entry |
//! | `W_PER_CMT_UNMATCHED`     | the other direction of the two above: a **declared** per-CMT entry — `[scaling] obs_scale[CMT=N]`, `[error_model] CMT=N:`, or a `y[CMT=N]` readout on either engine — that no observation is recorded on, so the entry is inert. A warning rather than an error because nothing goes `NaN`, and one code for all three channels, with the channel named in the message. Names `[data_selection]` as a candidate cause only for the dead compartments a clause actually emptied. Silent on an observation-free population, so `ferx check` without `--data` does not report every declared entry as dead (#1405) |
//! | `E_ENDPOINT_UNROUTED`     | a CMT declared as a non-Gaussian endpoint carries Gaussian observations — the population was read without the model's endpoint routing (#1199) |
//! | `E_ENDPOINT_NO_RECORDS`   | a declared non-Gaussian endpoint has no row routed to it (typically a missing `CMT` column) |
//! | `E_NO_SCORED_OBSERVATIONS` | the population has no record the likelihood scores — no Gaussian observation and no endpoint record on any subject — so the data contribute nothing to the objective. One population-level finding, last in `check_model_data_rule` so the specific data codes win `fit()`'s first error; the message names `[data_selection]` when it removed observation rows, and otherwise says what makes a row scored (#1491) |
//! | `E_DATA`                  | the `--data` file could not be read or parsed. Also raised by `run_sir` / `run_covariance` when the re-read of `fit.data_path` fails for any other reason than a covariate column (#1774), through the same classifier; there the message is the reader's own, without `ferx check`'s `Failed to read data file` lead |
//! | `E_SDE_INCOMPATIBLE`      | an SDE (`[diffusion]`) model used with SAEM / GN |
//! | `E_AD_RETIRED`            | `gradient_method = ad` requested; the Enzyme AD path was retired (use `auto` / `fd`) |
//! | `W_AUTO_OPTIMIZER_FOLLOWS_GRADIENT` | `gradient = fd` with `optimizer` left at `auto` on a model whose analytic gradient *is* in scope: `auto` follows the gradient, so the one line moved the optimizer too (#1381) |
//! | `E_IMP_CHAIN`             | `imp` mis-placed in a method chain (repeated / non-terminal) |
//! | `E_SAEM_NO_RANDOM_EFFECTS`| `method = saem` anywhere in a chain on a model with `n_eta = 0` |
//! | `E_METHOD_NO_RANDOM_EFFECTS` | `method = imp` / `impmap` / `bayes` anywhere in a chain on a model with `n_eta = 0` |
//! | `W_GN_NO_RANDOM_EFFECTS`  | `method = gn` as the last estimating stage on a model with `n_eta = 0` (start-sensitive; prefer `gn_hybrid`) |
//! | `E_OPTIMIZER_IOV`         | `optimizer = trust_region` used with an IOV model |
//! | `E_OPTIMIZER_AGQ`         | `optimizer = trust_region` used with a quadrature stage (`laplace`, or `focei` with `n_agq > 1`) |
//! | `E_SIGMA_ORDER_MISMATCH`  | a single-endpoint `[error_model]` names its sigmas in an order other than the `[parameters]` declaration order |
//! | `E_BLOCK_VARIANCE_ONLY`   | a scale tag (`(sd)` / `(variance)` / `(var)`) on a `block_omega` / `block_sigma` / `block_kappa` declaration, whose lower triangle mixes variances and covariances and so takes no single scale. The repair depends on which tag was written and rides along in `suggestion`: an `(sd)` author squares each SD into a variance and writes the off-diagonals as covariances; a `(variance)` / `(var)` author deletes the tag, which claimed nothing the lower triangle did not already say (#1377) |
//! | `E_OMEGA_INIT_AT_RAIL`    | a **free** `omega` / `kappa` / `[mixture] omega(k)` variance whose initial value packs onto the optimizer's `-6` lower rail (variance ≤ 6.1e-6, `~ 0.0` included) — clamped there and not estimable; `FIX` it or start it higher (#1229) |
//! | `E_THETA_INIT_OUTSIDE_BOUNDS` | a `theta` whose initial value is strictly outside its **own declared** range, compared on the declared scale so a lower bound of `0` is not confused with ferx's `1e-10` packing floor — clamped into the box and fitted from there; NM-TRAN refuses the identical stream (error 24) (#1251) |
//! | `E_INIT_BOUNDS_INVERTED` | a coordinate whose packed box is **empty** — for a `theta`, bounds swapped or a declared range lying entirely outside ferx's `1e-10` / `1e9` packing caps. No start can be placed in it, and the clamp has no interval to clamp into. Only θ reaches it through the parser, but the code is kind-neutral. The only start-side check with no `maxiter = 0` exemption (#1251) |
//! | `W_INIT_OUTSIDE_BOUNDS`   | an initial estimate strictly outside one of ferx's **internal** rails (the hidden `1e9` θ cap, the Ω `±6` / off-diagonal `±10` guards, the Σ `[-8, 5]` guard) — clamped there before the first objective evaluation (#1251) |
//! | `W_STEADY_STATE_II`       | SS=1 dose with missing / non-positive II |
//! | `W_STEADY_STATE_INFUSION` | SS=1 infusion with `T_inf > II` (overlapping pulses) |
//! | `W_STEADY_STATE_ABSOLUTE_TIME` | SS=1 dose on an `[odes]` PK block reading an absolute clock (`TAFD`, or `T` / `TIME`) — the run-in expands the train on a cycle-local clock, so there is no periodic limit to converge to: `TAFD` reads `NaN`, `T` / `TIME` return NONMEM's value (#1139) |
//! | `W_SDE_RESET`             | EVID=3/4 resets under an SDE model are not honoured |
//! | `W_SDE_LAGTIME`           | an absorption lag time under an SDE model is not honoured |
//! | `W_EXPERIMENTAL_SDE`      | an SDE (`[diffusion]`) model uses an experimental feature; the filter is covariance-only, so the state mean is never corrected by the data (see Feature Maturity and the SDE page) |
//! | `W_EXPERIMENTAL_NN`       | a neural-network (`[covariate_nn]`) model uses an experimental feature (see Feature Maturity docs) |
//! | `W_NN_BARE_OUTPUT_START`  | a `[covariate_nn]` block starts with every output-layer bias at 0 over live output weights — the start when it declares no `init` — so each output begins at `activation(0)` whatever parameter it feeds. Read from the start `fit()` is handed, so a warm start stays quiet (#1561) |
//! | `W_NEGATIVE_LAGTIME`      | `predict()` / `simulate()` only: a lag time is negative at the typical values, and is served — the non-fit severity of `E_NEGATIVE_LAGTIME` (#1189) |
//! | `E_DERIVED_NAME_CONFLICT` | `[derived]` name clashes with a built-in sdtab column, theta, eta, or indiv-param name |
//! | `W_DERIVED_COVARIATE_SHADOW` | `[derived]` name shadows a covariate (allowed but may be confusing) |
//! | `W_DERIVED_STEP_IGNORED`  | `step=` given for a DV-based integral (ignored; DV integrals use observation times) |
//! | `W_COV_ANALYTIC_SALVAGE` | the covariance step assembled the exact analytic R-matrix for the in-scope subjects and finite-differenced the information terms of the subjects outside its scope, rather than dropping the whole population onto the objective stencil (#1514) |
//! | `W_COV_QUADRATURE_OBJECTIVE` | a `saem` / `imp` / `impmap` estimate on a model with a non-Gaussian endpoint had its covariance R and S matrices taken on the 7-node adaptive Gauss–Hermite marginal (`method = laplace, n_agq = 7`) rather than the FOCEI objective, whose minimum can sit far from that estimate (#1781) |
//! | `W_COV_QUADRATURE_DECLINED` | the same fit where the quadrature marginal could not be used — a mixture, inter-occasion variability, or more than three random effects — so its standard errors are the FOCEI objective's curvature at a point that objective need not minimise (#1781) |
//! | `W_COV_NN_PENALTY` | a regularized `[covariate_nn]` fit's standard errors: under `covariance_method = r` / `rsr` they include the `nn_l2` / `nn_smooth` penalty's curvature (penalized-ML SEs); under `s` they cannot, and the note says so (#1352) |
//! | `W_COV_RSR_ONE_SUBJECT_LEVELS` | a successful `rsr` step on a θ level block, profiled or not, some levels of which a single subject reads: their `rsr` SEs are not usable, the message counts them per block (#2108 review row 3) |
//! | `W_COV_PROFILED_LEVELS_RESOLVED` | `run_covariance` on a profiled fit handed level estimates that are not the profile solve's (edited after the fit): the step's re-solve moved them, the reported estimates and SEs are the re-solved levels the matrix is at, and the message names how many moved and the largest move (#2419) |
//! | `W_COV_NON_STATIONARY_ESTIMATE` | the covariance R matrix needed an eigenvalue floor (indefinite, or an eigenvalue under `max(1e-10·λ_max, 1e-12)`) at a `saem` / `imp` / `impmap` / `vi` estimate, which those methods never minimise the R objective at; a requested `r` / `rsr` was replaced by `S⁻¹`, or the step failed when `S` was unavailable too (#1781) |
//! | `W_COV_ODE_TOLERANCE_NOT_TIGHTENED` | the covariance step's finite-difference matrices (`R`'s stencil, the score matrix `S`) of an integrated model could not be run at `ode_reltol = 1e-6` / `ode_abstol = 1e-8` and ran at the fit's looser tolerance, so the standard errors may be inflated by integration noise (#520) |
//! | `W_PROFILE_FD_GRADIENT` | a profiled fit on the analytic gradient route where the adjoint could not differentiate some subjects' terms over their profiled levels, in the fit's profile solves or in the covariance step's; those subjects' level gradients used reconverged central differences, re-solving their EBEs at every point. The levels are the same minimum, the solve slower (#2405) |
//! | `W_ABSORPTION_TWIN_DECLINED` | an analytic transit / IG model's ODE twin could not be built; the model stays closed-form with no ODE fallback (#1008) |
//! | `E_OUTPUT_UNKNOWN_COLUMN` | a name in `[output]` is not recognised as any known quantity |
//! | `W_OUTPUT_DUPLICATE`      | a name in `[output]` is already in the mandatory sdtab minimum |
//! | `W_ADDL_MISSING_II`       | ADDL > 0 on a dose row but II is zero or missing; additional doses not expanded |
//! | `W_COMPARTMENT_FREE_DOSES` | dose records in the dataset of a compartment-free (`$PRED`-equivalent) model, which applies no dose — reported once with the event (post-`ADDL`) and dosed-subject counts; the dose-level checks that read a topology (`E_MODELED_*_NO_PARAM`, `E_DOSE_CMT_*`) are skipped for this class, so the rows land here (#1443) |
//! | `W_MISSING_DV`            | EVID=0 observation row with a missing DV and no MDV=1; skipped rather than scored as DV=0 |
//! | `W_COVARIATE_MISSING`     | a covariate column that exists has no value (every cell `.`, blank or `NA`) for some subjects, so it is `NaN` for them. Reported once per covariate, naming it and up to five subject IDs. A relation that reads it is then `NaN` (non-finite objective) or, through `log`, `sqrt`, `min`, `max` or a comparison, a finite value with no further warning; guard it with `present(COV)` (#2041) |
//! | `W_IOV_OCC_MISSING`       | rows in the IOV occasion column had a missing or unparseable value and were assigned occasion 0 |
//! | `W_CENS_UNEXPECTED`       | an observation row's `CENS` cell is a whole number other than `-1` (above ULOQ), `0` (quantified) or `1` (below LLOQ), such as `7` or `-2`. Every engine reads only the flag's sign, so under `bloq_method = m3` a positive flag is scored on the lower tail, like `1`, and a negative one on the upper tail, like `-1`; under `bloq_method = drop` the row is scored as an ordinary observation. The message quotes the cell as written (`200`, not the `127` it saturates to) and is reported once per subject for each sign. It is not raised on a simulation design row with no `DV`, which is never scored. A cell that is **not a whole number** (`1.5`, `abc`, `inf`) is not this warning but a read error — `E_DATA` in `ferx check` — on a Gaussian observation row that carries a `DV` and that `[data_selection]` keeps; on a dose row, a row whose `DV` is missing, or a row the filter removes, the cell is not read (#1496) |
//! | `W_FILTER_COLUMN_ABSENT`  | a `[data_selection]` condition names a column the dataset does not have; the comparison never matches, so the condition does not select what was meant. The message names the missing column(s) and lists the headers the file does have. Raised once per read, whichever key the condition came from — and what "never matches" costs depends on that key: an `ignore` excludes nothing, an `accept` excludes everything. Reaches `ferx check` only since [#1465](https://github.com/FeRx-NLME/ferx-core/issues/1465), which made the check compile the clauses at all |
//! | `W_AMT_NOT_DOSED`         | record(s) carry `AMT != 0` but were **not** read as dose events, because `EVID` is neither 1 nor 4; their `AMT` was ignored. With no `EVID` column a nonzero `AMT` is enough to infer a dose, so this is a dataset that has one. Reported with the record and subject counts, and it wins over `W_NO_DOSES`, which is the generic backstop for a dataset carrying no `AMT` signal at all (#262) |
//! | `W_NO_DOSES`              | zero dose events parsed across every subject **although scored observations are present**. Silent on a dataset with no scored observations (an all-`EVID=2` / covariate-only file is not a fit), and silent when any subject carries a non-Gaussian observation — a TTE / survival or discrete endpoint legitimately has no PK dose, so warning there would be a false positive (#262) |
//! | `W_ALL_DOSES_ZERO`        | an `AMT` column is present and at least one dose event was parsed, but every parsed dose amount is exactly zero — so no drug enters the system and the objective is flat: the parameters will not move from their initial values. A warning rather than an error because the column is there and a genuinely dose-free-but-`AMT`-columned dataset is conceivable ([#753](https://github.com/FeRx-NLME/ferx-core/issues/753)) |
//! | `W_CMT_DEFAULTED`         | dose / observation rows assigned compartment 1 because the dataset has no `CMT` column, or the cell is missing or unparseable; reported when `CMT` selects something — more than one compartment a dose can reach (multi-state `[odes]`, or an analytical model whose `CMT=2` is a real target), a per-CMT scaling / error model / readout on either engine, an endpoint the row routes to, or a `[data_selection]` clause comparing `CMT` (the scope is the `CmtConsumer` enumeration in `api::validation`) |
//! | `E_COVSTAT_UNRESOLVED`    | a `[covariate_model]` relation still needs data-derived statistics (`center = median`, `levels = auto`, or a form whose default bounds come from the data) |
//! | `W_COVSTAT_UNBOUND`       | the same, reported without a `--data` file — the model is fine, it just cannot be built until a dataset is supplied |
//! | `E_COVARIATE_STATS_BINDING` | a `[covariate_model]` statistic could not be bound, either on the data (a `levels = auto` relation whose data carry one level, which leaves nothing to estimate, or a covariate with no non-missing value) or from a fit (statistics that lack a covariate a symbolic relation reads, carry one no relation reads, or a fit that recorded none on a model with no level block). `ferx check --data` reports the first kind; `bind_covariate_stats`, `bind_from_fit` and `layout_from_fit` return both, and `run_sir` / `run_covariance` return the second, since they call the from-fit binder, and the first kind's no-value refusal when they check a supplied model's statistics against the population of a fit that recorded none (#1729, #1774). Block `covariate_model`; the message is the binder's own (#1739, #1773) |
//! | `E_THETA_LEVEL_BINDING`   | a `[parameters]` level block (`theta NAME[COL, ...]`) could not be bound, either to the data's own levels (a level column missing or non-finite, no observation rows, a contrast the data cannot carry) or to a fit's level layout (a data level the fit estimated no θ for, level bindings that do not match the model's blocks, or a fit that recorded none). `ferx check --data` reports the first kind; the binders (`bind_theta_levels`, `bind_from_fit`, `layout_from_fit`, `bind_theta_levels_from_fit`) return both, and so do `run_sir` / `run_covariance`, which call them. A fit with no bindings, on a model with a level block and a symbolic statistic, is this code. Block `parameters` (#1773) |
//! | `E_THETA_LEVELS_DATA_UNBOUND` | a model whose `theta NAME[...]` block is bound was handed a population never bound for it, so no record carries an index into the block's levels: the usual "fit, then predict or simulate new data" without binding the new data. Raised by `fit()`, `predict` / `predict_diag`, the `simulate*` and adaptive entry points and `compute_npde_npd`. The binder named is `bind_from_fit` on the paths that run a θ already laid out, and `bind_theta_levels` on a fresh parse for `fit()`; the engine-internal index column is never named (#1647). Also raised by `predict_survival` / `predict_categorical` (#1763). A population only some of whose subjects carry the index gets "`k` of `n` subjects" and the first such ID instead of "never bound" (#1762); "never bound" is decided from the subjects, whatever `covariate_names` lists (#1795). Also raised by `run_sir` / `run_covariance` on a fit with no current population fingerprint (#1793) |
//! | `E_THETA_LEVELS_DATA_MISMATCH` | a population that carries a bound block's index was bound for other levels than the model's (#1762): a record's index is not the position, in the model's level table, of the label its own level columns spell — a population bound on its own for another dataset's studies or times, or a subset of the model's levels with its own count. Every unseen label is listed; or the first misindexed record (subject, time, label, both indices) is named, with the same binder as `E_THETA_LEVELS_DATA_UNBOUND`; or an index without its level column is named, since its level cannot be checked. Raised by the same entry points, and by `run_sir` / `run_covariance` on a fit with no current population fingerprint (#1793; with one, `E_POPULATION_MISMATCH` instead), where unseen labels are told to pass the population the fit was given. Refused, never re-indexed |
//! | `E_MODEL_READ`            | the model file could not be read: it does not exist, cannot be opened, or is not UTF-8. Raised by `ferx check` before any parse; the message names the path and the operating system's reason. No block, since no block is at fault. The file is read once per check, and that one read is parsed and bound to the data, so a file moved or edited while the check runs is not read again (#1743, #1752). `run_sir` / `run_covariance` raise it too when they re-read `fit.model_path`, through the same classifier, and there a file edited since the fit is `E_FIT_INPUT_CHANGED` (#1774) |
//! | `E_FIT_OPTION`            | a call-time override handed to `validate_model_file_with` (#2165) names no fit option, or a value its key does not accept. No block; terminal, like a parse failure. A model file's own `[fit_options]` refuses the same pair at parse time as `E_PARSE` |
//! | `E_COV_LEVEL_UNKNOWN`     | the data carry a value of a categorical `[covariate_model]` covariate that is not one of the relation's levels, which the generated factor would model as the reference. Raised by `fit()`, `ferx check`, the `simulate*` and adaptive entry points, `predict` / `predict_diag` / `predict_survival` / `predict_categorical`, `compute_npde_npd`, and `run_sir` / `run_covariance` (#1111, #1740); not by the per-method estimators `fit()` dispatches to, which rely on its check |
//! | `E_PROFILE_SCOPE`         | a `profile`d θ level block (#1065) that the profile solve cannot fit: `profile` on a single θ, with `FIX`, or with a point box, an empty box or an init outside its box (#2112) (raised at parse time, so a parse error's message carries the code); or, from `check_model_options` (so `fit()` and `ferx check`), a method other than `foce` / `focei` or `n_agq > 1`, a non-Gaussian endpoint, a non-additive or log-transformed residual error, M3 censoring, a `[mixture]` / `[diffusion]` / FREM model, `optimizer = trust_region` / `mma`, `global_search`, a parameter prior. Never a fallback. Block `parameters` |
//! | `E_PROFILE_NOT_LINEAR`    | a `profile`d block whose prediction is not `c × block` plus terms that do not read it, with a data-only `c` (#1065): read through a function, power or remainder; through an inline `if` or an `if` branch whose condition reads more than data; in a product or quotient of two reads; scaled by a parameter or random effect; as a divisor; through the states; by the residual error; beside a second profiled block; or by no prediction at all. Names the construct and the site. Raised at parse time. Block `parameters` |
//! | `E_THETA_LENGTH`          | an entry point taking a parameter vector was handed a θ whose length is not the model's; refused rather than read by position (a short vector read `0.0` past its end). Raised by the `simulate*` and adaptive entry points (#1614), `predict` / `predict_diag` / `predict_survival` / `predict_categorical` and `compute_npde_npd` (#1615); never by `ferx check`, which has no parameter vector. On a level-block model the message says the θ count is set by the bound data, so a fit's θ fits only a design bound against that fit's level bindings (#1614); it names no function, since the R wrapper reaches it too (#1623). `fit()` refuses its initial parameters with the same message (#1764); and `run_sir` / `run_covariance` refuse a fit whose θ is not the length of the model they run on, with their own message (#1774) |
//! | `E_PARAM_SHAPE`           | an entry point was handed an Ω, σ or Ω_IOV whose dimension is not the model's, an Ω_IOV on a model with no κ, or none on a model with κ; refused rather than read by position (a mis-sized block panicked; a long σ or a missing Ω_IOV was silently ignored). Raised by `fit()` (as its message; and for `init_params` with no per-class Ω/Σ on a `[mixture]` model, #1788), the `simulate*` and adaptive entry points, `simulate_with_uncertainty` and `compute_npde_npd`, and — for σ and Ω_IOV — `run_sir` / `run_covariance` (#1764); for Ω_IOV alone, `fitted_params_from_result` (#1789); and inside the SIR core for per-class Ω/Σ present on parameters for a model with no `[mixture]`, or absent on one with it (#2044), which no public entry point reaches with a code: `run_sir` builds the parameters from the fit, so their per-class Ω/Σ follow the model, and `run_sir_core` returns the text as a `String`. Not by `predict*`, which read none of the three; on `simulate*` a missing Ω_IOV keeps its own message (#1019) under this code (#1774), and `compute_npde_npd` falls back to κ = 0 for one, as documented there |
//! | `E_POPULATION_MISMATCH`   | a population handed to, or re-read by, a post-hoc step is not the one the fit was given (#1685): another subject list, other covariate columns, or a subject whose records, doses or covariate values differ. Raised by `run_sir` / `run_covariance` and `PopulationFingerprint::verify` (#1813, #1814), from one builder, so the three give the same message and suggestion. The message names what differs and how the population was obtained, never a cause the comparison cannot establish; the suggestion is the remedy. Never by `ferx check`, which has no fit. Pre-empted by `E_ENDPOINT_UNROUTED` and `E_THETA_LEVELS_DATA_UNBOUND`, whose fix is a reader or a binder |
//! | `E_FIT_SUBJECTS_MISMATCH` | the population matches the fit's fingerprint, so it is the one the fit was given, but the fit's per-subject results (`fit.subjects`), which `run_sir` / `run_covariance` read by position, are another count or another order. Reachable when a front end rebuilds a fit from separate fields (ferx-r's fit skeleton); a `fit()` result carries both lists from one population. Without a current fingerprint the same mismatch is `E_POPULATION_MISMATCH`, since nothing says which side is wrong (#1868) |
//! | `E_FIT_INPUT_MISSING`     | a post-hoc step on a fit (`run_sir`, `run_covariance`) needs an input that neither the caller passed nor the fit records: `fit.model_path` with no model passed, `fit.data_path` with no population passed, SIR's `fit.covariance_matrix`, the population of an IOV model passed as `model = Some` to a fit with no recorded reader settings, the IOV occasion rule of a fit saved before #1783 whose population carries no occasion labels, or the fitted values of a `[mixture]` fit's per-class overrides, on a fit that records neither `mixture_overrides` nor a `packed_estimate` (#1765; `simulate_with_uncertainty` raises this one too). `simulate_with_uncertainty` also raises it for the covariance matrix or SIR resamples it draws from, absent or empty (#2044). No block. Never by `ferx check`, which has no fit (#1774) |
//! | `E_FIT_INPUT_CHANGED`     | a file a post-hoc step re-reads, `fit.model_path` or `fit.data_path`, no longer has the SHA-256 the fit recorded. No block. Raised by `run_sir` / `run_covariance` (#1774); never by `ferx check`, which compares no hash |
//! | `E_FIT_INPUT_MODEL`       | the model a post-hoc step would run on is not the one the fit was made with: a supplied model bound with other data-derived bindings than `fit.data_bindings`, a supplied model whose covariate statistics the population does not reproduce (on a fit that recorded no bindings, #1729), a model whose η count is not the width of the fit's per-subject η̂, a `[mixture]` fit whose recorded `mixture_overrides` do not hold one non-negative finite value per declared override (#1765), or which records none and whose `packed_estimate` has another layout or no longer reproduces its reported estimates (#1815; `simulate_with_uncertainty` raises these too), or a covariance matrix that is not the size of the model's packed parameters (SIR's proposal; and `simulate_with_uncertainty`'s covariance or SIR resamples, #2044). No block. Raised by `run_sir` / `run_covariance` (#1774). Its θ count is `E_THETA_LENGTH`, its Ω / σ / Ω_IOV shape `E_PARAM_SHAPE` |
//! | `E_SIR_OPTIONS`           | SIR settings the importance resampler cannot run with: `sir_resamples` above `sir_samples`, a `sir_df` that is not a valid Student-t degrees of freedom, or `sir_scale = natural` on a model with `prior(...)` (#1723). A `sir_df` below 1 is refused although `ChiSquared` would run it, since `[fit_options]` and the `.fitrx` reader refuse it (#1770). No block. Raised by `run_sir` (#1774), and by `fit()` / `ferx check` for a `sir_df` below 1 when `sir = true` or `covariance_fallback = sir` (#1770); otherwise `fit()`'s in-fit SIR reports the same text as a `SIR failed:` warning, and `run_sir_core` returns it as a `String` |
//! | `E_SIR_NOTHING_TO_SAMPLE` | every estimated parameter of the fit is FIX (θ, Ω and σ, and Ω_IOV, `block_sigma` correlations and `[mixture]` overrides where the model has them), so SIR has no free direction to propose in. Block `parameters`. Raised by `run_sir` (#1774); `fit()`'s in-fit SIR reports the same text as a `SIR failed:` warning, and `run_sir_core` returns it as a `String` |
//! | `E_ODE_SOLVER_OPTIONS`    | ODE solver settings on a hand-built `FitOptions` that no solver can run with: an `ode_reltol` / `ode_abstol` that is not finite and positive, `ode_max_steps = 0`, or `ode_stiff_abort_after = Some(0)`, which `[fit_options]` spells as off (`None`) and a `.fitrx` reader refuses (#1770). Refused whether or not the model integrates an ODE, since every fit records them. No block. Raised by `run_sir` / `run_covariance` through `with_fit_scope` (#2044) and by `fit()` through `install_on_fit_pool` (#1772), and `run_sir_core` returns the same text as a `String`; a model file's `[fit_options]` refuses the same values at parse time as `E_PARSE` |
//! | `E_LOGIT_THETA_OUT_OF_RANGE` | a free θ used as `inv_logit(logit(θ) + ETA)` whose estimate is not strictly inside (0, 1), so it has no logit-scale draw; reachable only with a declared upper bound above 1, which the suggestion says to lower. Block `parameters`. Raised by `simulate_with_uncertainty` (asymptotic method) and `run_sir` (#2044). `Display` folds the suggestion in after the message, so `run_sir_core` and `draw_parameter_samples` return that text as a `String`, and `fit()`'s in-fit SIR reports it as a `SIR failed:` warning |
//! | `E_RTTE_RECORDS`          | a repeated-event (`type = rtte`) endpoint's records cannot be scored: interval-censored, a non-finite `TIME`, an entry time after the record's `TIME`, out of time order, or (clock reset) a right-censored row before a later record or two events at one `TIME`. Block `event_model`. Raised by `fit()` and `ferx check --data` (#1788); the reader drops some of these on load, so mostly a hand-built `Population` reaches it |
//! | `E_BINARY_STATE`          | a `[binary_model]` record whose observed DV is not 0 or 1 (#760). Block `binary_model`. Raised by `fit()` and `ferx check --data` (#1788) |
//! | `E_CTMM_RECORDS`          | a `[markov_model]` (CTMM) record whose observed DV is not one of the declared `states`, or a subject whose CTMM observation times decrease (the reader sorts doses, not observations) (#759). Block `markov_model`. Raised by `fit()` and `ferx check --data` (#1788) |
//! | `E_LTBS_UNSUPPORTED`      | LTBS (`log_transform = true`) with a proportional or combined error, a per-CMT error spec, or an SDE. Block `error_model`. Only a hand-built `CompiledModel` reaches it; the parser refuses all three. Raised by `fit()` and `check_model_options` (#1788) |
//! | `E_IIV_ON_RUV_SHARED_ETA` | the `iiv_on_ruv` η is also used in `[individual_parameters]`; the residual-error random effect must be a dedicated omega (#409). Block `error_model`. Raised by `fit()` and `ferx check` (#1788) |
//! | `E_IIV_ON_RUV_METHOD`     | `iiv_on_ruv` with a non-interaction stage (`foce` / `gn` without `interaction`) anywhere in the method chain; `Y = IPRED + EPS·EXP(ETA)` is an interaction model (#409). Block `fit_options`. Raised by `fit()` and `ferx check` (#1788) |
//! | `E_MIXTURE_METHOD`        | a `[mixture]` model with a method its objective is not wired for: FOCEI quadrature (`n_agq > 1`), an estimator other than FOCE / FOCEI / SAEM / Bayes / IMP / IMPMAP, or Bayes with per-class `omega(k)` / `sigma(k)` overrides (#977, #985). Block `fit_options`. Raised by `fit()` (on its `init_params`) and `ferx check` (on the parsed parameters) (#1788) |
//! | `E_ENDPOINT_TV_COVARIATE` | a covariate that varies within a subject is read by an endpoint that evaluates it at the baseline: an analytic hazard's own parameters, any covariate under an ODE-accumulated hazard, a `[binary_model]` predictor, a `[markov_model]` intensity (#741). Block is the endpoint's (`event_model` / `binary_model` / `markov_model`). Raised by `fit()`, `ferx check` (since #1774), `predict` / `predict_diag` / `predict_survival` / `predict_categorical` and the `simulate*` entry points |
//! | `E_ENDPOINT_TIME_DEPENDENT_PK` | an ODE-accumulated hazard would integrate a `TIME`-dependent individual parameter (reaching it through the `[odes]` RHS or an `init()` seed) at its TIME=0 value (#1853), or, fail-closed, a hand-built model lacks the compiled programs that decide it. Block `event_model`. A function of the model alone. Raised by `fit()`, `ferx check` (without `--data`, since #1774), `predict` / `predict_diag` / `predict_survival` and the `simulate*` entry points; not `predict_categorical` |
//! | `E_READOUT_DEPOT_UNSUPPORTED` | an analytic `[scaling] y` readout reads the oral `depot` amount on a subject dose superposition cannot reconstruct it for: an EVID=3/4 reset (#650) or a dose into a non-default compartment (#375). Block `scaling`. Raised by `fit()`, `ferx check` (since #1774) and `predict` / `predict_diag` |
//! | `E_CTMM_UNSUPPORTED`      | a `[markov_model]` endpoint reached `predict` / `predict_diag` with no continuous observation, or a `simulate*` entry point; neither has an occupancy path (#759, #820). Block `markov_model`. Never by `ferx check` or `fit()`, which supports CTMM (#1774); `run_model_simulate` raises it for a `[simulation]` block on a CTMM model (#2062) |
//! | `E_TTE_NOT_SIMULATABLE`   | a TTE endpoint the event-time simulator cannot draw for this population: two repeated-event streams on a subject, a repeated-event cause beside a single-event one, a reset under a repeated-event or ODE-accumulated hazard, left truncation under an ODE-accumulated hazard; or a repeated-event or ODE-accumulated TTE endpoint on `simulate_with_uncertainty`, which needs a horizon and takes none. Block `event_model`. Raised by the `simulate*` entry points (#1774) |
//! | `E_SIMULATE_HORIZON`      | the simulation horizon is missing where a repeated-event or ODE-accumulated TTE endpoint needs one (on an entry point that takes one; `simulate_with_uncertainty` gives `E_TTE_NOT_SIMULATABLE` instead), is not finite and positive, or lies below a subject's entry time. Raised by the `simulate*` entry points (#1774), and by `run_model_simulate` for a `[simulation]` block with a TTE endpoint of any kind and no `horizon`, the censoring time of its template rows (#1788) |
//! | `E_SIMULATION_DESIGN`     | a `[simulation]` design that gives the model nothing to observe: no `times` for a continuous (residual-error) or `[binary_model]` endpoint, or no `times` and no TTE endpoint to observe at a `horizon`. Block `simulation`. Raised by `run_model_simulate` (#1788); `ferx check` does not read `[simulation]` |
//! | `E_DATA_PATH_MISSING`     | a file entry point (`fit_from_files`, `prepare_run`, `run_model_with_data` and their variants) was given no data path and the model file has no `[data]` block. No block. `ferx check` skips its data checks there instead of refusing, so only the entry points raise it (#1788); `resolve_data_path` returns the same text as a `String` |
//! | `E_PROPENSITY_INPUT`      | propensity-score matching was requested on a population it cannot compute posthoc etas from: empty, a subject with no observations, a non-finite `DV`. Raised by `simulate_with_options*` (#1774) |
//! | `E_ADAPTIVE_UNSUPPORTED`  | a model the adaptive driver cannot simulate faithfully (no `[odes]`, an SDE, a covariate-selected `[error_model]`, a residual `correlation` line, a log-transformed (LTBS) error model — #2129), an `auc_target` on a time-varying-covariate, IOV or reset subject, or a base regimen × EVID=3/4 reset under a time-varying covariate or IOV (before any subject runs). Once the driver runs (#1983): a base dose under a time-varying covariate or IOV that is not a plain bolus / infusion, a non-finite break time, a `TAD` / `TAFD` read before any dose. Block when one is at fault. Raised by `simulate_adaptive` / `simulate_adaptive_from_spec` (#1774) |
//! | `E_ADAPTIVE_OPTIONS`      | `AdaptiveSimulateOptions` with no usable schedule (`decision_times` empty, not finite, not strictly increasing, or past `max_decisions`), a monitor compartment out of range, a `Dv` monitor on a compartment no `[error_model]` covers, or `decision_times` / `monitors` set beside a spec that owns them — all before any subject runs. Once the driver runs (#1983): two of a subject's times within 1e-15 of each other but not equal. Raised by the adaptive entry points (#1774) |
//! | `E_ADAPTIVE_CONTROLLER_ACTION` | a controller action the driver refuses: a dose into compartment 0 or past the states, a negative or non-finite amount, a rate that is not positive and finite, a compartment fed by an input-rate function or carrying a lag, a `Stop` with actions after it. The controller is the caller's code, so an input refusal; only a simulation reaches it. Raised by `simulate_adaptive` (#1983); on `simulate_adaptive_from_spec`, whose controller is the spec's `route`, the same refusal is `E_ADAPTIVE_SPEC`. A non-finite amount computed from a diverged ODE state is not this code: the driver refuses the state (or a non-finite monitor reading) before the controller runs, uncoded (#1578) |
//! | `E_ADAPTIVE_SPEC`         | an `[adaptive_dosing]` spec that does not compile against the model: a parser invariant (on a spec built in code), `assay_cmt` out of range or with no `[error_model]` (#1983), a `route` compartment out of range or fed by an input-rate function, or — once the driver runs — a `route` dose it refuses (a lagged compartment) (#1983), an `observe` that does not compile. Block `adaptive_dosing`. Raised by `simulate_adaptive_from_spec`, whose context it carries (#1774). An `observe` covariate absent from the data is `E_MISSING_COVARIATE` there, block `adaptive_dosing` |
//! | `E_SUPPLIED_ETAS`         | per-subject η that an adaptive run cannot use as given (#1594): from `SuppliedEtas::from_table` / `from_fit`, IDs and rows that differ in count, a row that is not one finite value per named η, a repeated ID, a fit whose η names are not the model's; from the `*_with_inputs` adaptive entry points before any subject runs, a `[mixture]` model (the run carries no class), η names that are not the model's in its order, or a simulated subject with no supplied η under `UnmatchedSubjects::Refuse`; from `predict_survival_at_etas_diag` after `predict_survival_diag`'s checks (#1885), the same three, a subject with no supplied η always refused |
//! | `E_LEVEL_MIXED_WITH_IOV_COLUMN` | a `level` declaration beside `iov_column` / `iov_occasion`, or a `kappa` with no `at LEVEL` once levels are declared (#1082). Block `parameters`; raised at parse time, so a parse error's message carries the code. Also a derived `iov_occasion` rule handed to `fit()` or `simulate_with_options*` with a nested model, block `fit_options` |
//! | `E_LEVEL_MISSING_LABEL`   | a record with no value in a level column of a nested model (#1082), including a blank cell a model-blind reader filled from another record (#2036). Block `parameters`. Raised by `fit()`, `ferx check --data` and every `simulate*` entry point |
//! | `E_LEVEL_SINGLE_MEMBER`   | a level with at most one group in every parent group whose kappa reads the same parameters as a random effect of the parent level (#1082). Block `parameters` |
//! | `W_LEVEL_SUMMARY` | informational (severity `Info`, so not counted as a warning, #2387): one row per level of a model with nested levels — its groups, the groups per parent (min / median / max), the parents that hold a single group, and its kappas, counted on the observation records (#2035). Raised by `ferx check --data` only. Block `parameters` |
//! | `E_NESTED_LEVEL_METHOD_UNSUPPORTED` | a nested model fitted with a stage other than `foce` / `focei` at `n_agq = 1`, or with a `[mixture]` / FREM (#1082). Block `fit_options` |
//! | `E_RESIDUAL_CORRELATION_GROUPING_AMBIGUOUS` | a residual `correlation … by` line on a nested model whose grouping a level and a data column would read two ways: a level in a list of columns, a level name that is also a different column, or a nested level's column under another name (#1082 PR5). Raised by `fit()`, `ferx check --data` and `simulate()`. Block `error_model` |
//! | `E_RESIDUAL_CORRELATION_CONFOUNDED_KAPPA` | a `correlation cs(…)` line whose blocks partition the rows exactly as a kappa-carrying level's groups (nested, or the occasions of one level): the kappa is the same covariance (#1082 PR5). Raised by `fit()` and `ferx check --data`. Block `error_model` |
//! | `W_RESIDUAL_CORRELATION_PARTLY_CONFOUNDED_KAPPA` | the same partitions agree in some subjects but not all: identified from the others, so it fits, with the count said (#1082 PR5). Raised by `fit()` and `ferx check --data` |
//! | `E_NESTED_LEVEL_SIMULATE_UNSUPPORTED` | a nested model reached an adaptive-dosing entry point, whose kappa stream draws one kappa per occasion (#1082); `simulate*` draws per level. Block `parameters`. Never by `ferx check` |
//! | `W_PARSE` | generic code for a parse-time warning with no narrower code (e.g. mu-referencing notes, `W_NN_WEIGHT_DIRECT_REFERENCE`); the block context is in the message text |
//! | `W_DATA_PATH_OVERRIDE` | an explicit data path (`--data`, R) and the model's `[data] path = ...` name different files; the explicit one is used (#690) |
//! | `E_PRIOR_UNRESOLVED` | a `prior(...)` (or `[priors] from_fit`) cannot be resolved against the parameter layout: unknown name, a FIXed parameter, a block element, a duplicate, a bad central value, or an unexpanded `from_fit` (#254) |
//! | `W_UNUSED_PARAM` | a theta, omega, kappa or sigma declared in `[parameters]` is not referenced by any model expression (sigma: by `[error_model]`), so it cannot affect predictions |
//! | `E_KAPPA_WEIGHT_NONPOSITIVE` | a kappa's `weight = W` (κ ~ N(0, Ω_IOV / W)) evaluates to zero, negative or non-finite at an observation or dose record, at the model's initial θ (#1031) |
//! | `W_KAPPA_WEIGHT_VARIES_WITHIN_OCCASION` | a kappa's sample-size weight is not constant across the observations of one occasion (or one group of its level under nested levels); reported once per kappa (#1031) |
//! | `W_ADDITIVE_INIT_SCALE` | a `combined` error model's additive SD initial estimate is below 1% of the median abs(DV) of the observations it covers, which can trap the fit in a collapsed-additive local minimum |
//! | `W_ZERO_DV_MAGNITUDE_ONLY` | observations with DV = 0 are scored by a residual error model whose variance is zero at a zero prediction (proportional, power, or an additive SD starting at 0), so their likelihood has no stable value near f = 0 and a fit can stop at a wrong optimum; fit and `ferx check` only, not simulate (#2459) |
//! | `W_DATA` | `ferx check` relay code for a data-reader warning whose message states no `W_` code of its own |
//! | `E_DOSE_NO_AMT` | the dataset has dose records (`EVID=1/4`) but no `AMT` column, so every dose amount would be zero (#753) |
//! | `E_DOSE_CMT_OUT_OF_RANGE` | a dose record (any `AMT`, bolus or infusion) whose `CMT` exceeds the model's compartment count — the `[odes]` states, or the analytical model's compartments (#899) |
//! | `E_DOSE_CMT_NOT_INFUSABLE` | an infusion with `AMT ≠ 0` into compartment 0 (`CMT=0`), which is a default bolus compartment with no zero-order target, on an ODE or event-walk analytical model (#899) |
//! | `W_STEADY_STATE_INIT` | subject(s) have SS=1 doses while the analytical model declares an `[initial_conditions]` baseline; the baseline is superposed on the steady state and double-counts the starting amount (#521) |
//! | `E_MODELED_DURATION_NO_PARAM` | a `RATE=-2` dose (modeled infusion duration) into compartment n on a model that declares no `D{n}` parameter; one per compartment. Not raised on a compartment-free model |
//! | `E_MODELED_RATE_NO_PARAM` | a `RATE=-1` dose (modeled infusion rate) into compartment n on a model that declares no `R{n}` parameter; one per compartment. Not raised on a compartment-free model |
//! | `W_MODELED_DURATION_NONPOSITIVE` | a modeled infusion duration `D{n}` (RATE=-2) is at or below its `1e-8` floor at a dose record at the initial typical values (η = 0); the dose is delivered as a bolus-like spike |
//! | `W_MODELED_RATE_NONPOSITIVE` | a modeled infusion rate `R{n}` (RATE=-1) is at or below its `1e-8` floor at a dose record at the initial typical values (η = 0); the dose trickles in over a near-infinite duration |
//! | `W_NN_WEIGHT_DIRECT_REFERENCE` | an expression reads a generated `[covariate_nn]` weight or bias θ directly; the analytic NN θ-gradient shortcut is disabled and every θ falls back to finite differences |
//! | `E_THETA_LEVELS_UNBOUND` | a `theta NAME[...]` level block reached an entry point without being bound to data, so it has no levels and gathers NaN (#1064) |
//! | `E_FREM_NON_GAUSSIAN_ENDPOINT` | FREM requested on a model that declares a non-Gaussian endpoint (`[event_model]` / `[binary_model]` / `[markov_model]`); FREM is Gaussian-only (#1199) |
//! | `E_THETA_GATHER_INDEX_MISSING` | a `theta NAME[COL]` block is indexed by a data column a subject's observation or dose record carries no value for (#1064) |
//! | `E_THETA_GATHER_INDEX_RANGE` | a `theta NAME[COL]` index column holds a value that is not a whole number in 1..=N for the block's N levels (often a 0-based column) (#1064) |
//! | `E_RESIDUAL_CORRELATION_RANGE` | a residual `correlation` line's ρ range does not keep R positive definite (`cs` at or below −1/(n−1), `car1` ρ < 0, abs(ρ) ≥ 1) or a free ρ's bound exceeds ±0.995 (#1081) |
//! | `E_RESIDUAL_CORRELATION_RESET` | an `ar1` / `car1` residual `correlation` block holds rows on both sides of an EVID=3/4 record that restarts TIME, so the time between them is undefined (#1081) |
//! | `E_RESIDUAL_CORRELATION_TIES` | an `ar1` / `car1` residual `correlation` block has two rows at the same TIME: `ar1` cannot order them and `car1` makes R singular (#1081) |
//! | `W_RESIDUAL_CORRELATION_INERT` | a residual `correlation … by …` line under which no block holds two observations in any subject, so ρ never enters the likelihood and stays at its initial value (#1081) |
//! | `W_BLOCK_SIGMA_L2_ORDER` | a `block_sigma` correlated residual whose co-temporal rows can pair more than one way, with no `L2` column — the pairing follows CSV row order (#830) |
//! | `W_L2_UNUSED` | the dataset has an `L2` column but the model declares no `block_sigma` correlated residual, so `L2` is inert and not available as a covariate (#830) |
//! | `E_AGQ_NODES_ZERO` | `n_agq = 0`, under any method: every fit records `n_agq`, and a `.fitrx` holding 0 does not load (#1770); use `n_agq = 1` for plain FOCEI / one-node Laplace |
//! | `E_AGQ_NODES_TOO_LARGE` | `n_agq` above the 21-node per-dimension limit: on a quadrature stage (`laplace`, or `focei` with `n_agq > 1`), and on any other chain too, where the value goes unused but is recorded (#1770) |
//! | `E_AGQ_GRID_TOO_LARGE` | a quadrature stage (`laplace`, or `focei` with `n_agq > 1`) whose tensor grid `n_agq^n_eta` exceeds 100,000 nodes per subject; under IOV the grid is `n_agq^(n_eta + K·n_kappa)` with `K` the most occasions any subject has, which only `fit()` and `ferx check --data` can count (#1788) |
//! | `E_AGQ_EVAL_ONLY` | `agq_eval_only = true` with no `laplace` stage in the method chain, or with `laplace` not the final stage |
//! | `E_FOCEI_NAGQ_UNSUPPORTED` | `method = focei` with `n_agq > 1` on a model outside the analytic sensitivity (adjoint) scope; use `laplace` with `n_agq > 1` instead (#251) |
//! | `E_VI_NAGQ_UNSUPPORTED` | `n_agq > 1` on a chain containing `method = vi` but no `laplace` or `focei` stage to consume it |
//! | `E_BINARY_GN_UNSUPPORTED` | `method = gn` / `gn_hybrid` anywhere in the chain on a model with a `[binary_model]` (binary / categorical) endpoint (#760) |
//! | `E_CTMM_GN_UNSUPPORTED` | `method = gn` / `gn_hybrid` anywhere in the chain on a model with a `[markov_model]` (CTMM) endpoint (#759) |
//! | `E_CTMM_IOV_UNSUPPORTED` | a `[markov_model]` (CTMM) endpoint on a model with IOV (`n_kappa > 0`); the generator uses BSV η only, so κ would be ignored (#759) |
//! | `E_IMP_MIXTURE_IOV_UNSUPPORTED` | `method = imp` / `impmap` anywhere in the chain on a `[mixture]` model with IOV (`n_kappa > 0`); their mixture E-step has no κ (#2301) |
//! | `E_VI_MODEL_UNSUPPORTED` | `method = vi` anywhere in a chain on a model with a non-Gaussian (TTE / categorical) endpoint or an SDE (`[diffusion]`) block, whose likelihood VI's fixed-η data term does not evaluate |
//! | `W_VI_OMEGA_UNANCHORED` | `vi_kl = mc` with `vi_omega_update = adam` on a chain that runs VI leaves Ω driven only by a Monte-Carlo gradient; expect a noisy Ω trace and possible non-convergence |
//! | `E_BLOCK_SIGMA_METHOD_UNSUPPORTED` | a `block_sigma` model with a stage other than `foce`, `focei`, `saem`, `imp`, `laplace` or `vi` in the chain (i.e. `gn`, `gn_hybrid`, `impmap`, `bayes`) |
//! | `E_BLOCK_SIGMA_CHAIN_UNSUPPORTED` | a `block_sigma` model with a free correlation whose chain runs a stage other than `foce`/`focei`/`laplace` after one of those; it would score the declared rho, not the estimate (#847) |
//! | `E_BLOCK_SIGMA_M3_UNSUPPORTED` | `block_sigma` correlated residual errors with `bloq_method = m3` (censored-observation likelihood) |
//! | `E_BLOCK_SIGMA_FREM_UNSUPPORTED` | `block_sigma` correlated residual errors on a FREM model (covariate pseudo-observations) |
//! | `E_BLOCK_SIGMA_IIV_ON_RUV_UNSUPPORTED` | `block_sigma` correlated residual errors together with `iiv_on_ruv` residual-error scaling |
//! | `E_BLOCK_SIGMA_IOV_UNSUPPORTED` | `block_sigma` correlated residual errors on a model with IOV (`n_kappa > 0`) under a method other than `foce` / `focei`, or with `n_agq > 1`; only the FOCE/FOCEI IOV marginal scores the full residual covariance (#979) |
//! | `E_BLOCK_SIGMA_SDE_UNSUPPORTED` | a `block_sigma` SDE (`[diffusion]`) model with `saem` in the chain; SAEM's data term carries no EKF process noise (FOCE/FOCEI are fine) |
//! | `E_BLOCK_SIGMA_TTE_UNSUPPORTED` | `block_sigma` correlated residual errors on a model with any non-Gaussian endpoint (TTE, binary/categorical, Markov), for every method |
//! | `E_RESIDUAL_CORRELATION_SCOPE` | a residual `correlation` line with a method other than `foce` / `focei`, `n_agq > 1`, `bloq_method = m3`, FREM, a non-Gaussian endpoint, a mixture or an SDE model (#1081) |
//! | `E_PRIOR_METHOD_UNSUPPORTED` | a parameter prior is in force but the last estimating stage is not `foce` / `focei` / `laplace` / `gn` / `gn_hybrid`, so the prior would be silently dropped (#254) |
//! | `E_PRIOR_COV_METHOD_UNSUPPORTED` | a parameter prior is in force with the covariance step on and `covariance_method = s` or `rsr`, neither of which can carry a prior's curvature (#254) |
//! | `E_ABSORPTION_DOMAIN` | a built-in absorption input-rate parameter (e.g. transit `mtt ≤ 0` / `n < 0`, igd `mat`/`cv2 ≤ 0`, a non-finite per-route `lag=`) is out of domain at typical values at some dose record |
//! | `E_ABSORPTION_FRACTION` | a pathway fraction (`FR*fn(...)`) on a built-in absorption term is outside (0, 1], or the fractions on one compartment do not sum to 1 (±1e-4), at typical values at some dose record (#388) |
//! | `E_ABSORPTION_RATE_ZERO_ORDER` | an infusion (`RATE>0`) into a compartment fed by a `zero_order()` absorption function; infusing a spanning window is not supported (#719) |
//! | `E_ABSORPTION_SS_INFUSION` | a steady-state infusion (`SS=1` with `RATE>0`, any `II`) into a compartment fed by a built-in absorption input-rate function (#719) |
//! | `E_ABSORPTION_SS_LAG` | an `SS=1` dose (`II > 0`) into a built-in absorption compartment that carries a lag (compartment `lagtime`/`ALAG` or a per-route `lag=`) (#719, #856) |
//! | `E_ABSORPTION_SS_ZERO_ORDER` | an `SS=1` dose (`II > 0`) into a compartment fed by a `zero_order()` absorption function; periodic window equilibration is not implemented (#719) |
//! | `E_ABSORPTION_DIFFUSION` | an `[odes]` model with a built-in absorption input-rate function (`transit()`, `igd()`, …) also has a `[diffusion]` block; the EKF propagation does not carry the input-rate forcing |
//! | `E_TRANSIT_UNSUPPORTED` | a transit closed form (`one_cpt_transit` / `two_cpt_transit`) meets a feature it cannot serve: a non-depot (CMT≥2) dose, an EVID=3/4 reset, an SS infusion, or (twin-less) IOV, a TIME-dependent parameter, TV covariates, SS or an infusion (#386) |
//! | `E_TRANSIT_FLIP_FLOP` | a twin-less transit closed form (`one_cpt_transit` / `two_cpt_transit` with no ODE twin) starts outside its convergence domain at η = 0 — flip-flop or (2-cpt) coincident eigenvalues — so it predicts zero (#776); also every uncertainty draw flip-flopped |
//! | `W_TRANSIT_FLIP_FLOP` | a twin-carrying transit absorption model is in the flip-flop regime at typical values (η = 0) for some subject; ferx evaluates the equivalent ODE model there (informational) |
//! | `E_IG_UNSUPPORTED` | an inverse-Gaussian closed form (`one_cpt_ig` / `two_cpt_ig`) meets a feature it cannot serve: a non-depot (CMT≥2) dose, an EVID=3/4 reset, an SS infusion, or (twin-less) IOV, a TIME-dependent parameter, TV covariates, SS or an infusion (#790) |
//! | `E_IG_FLIP_FLOP` | a twin-less inverse-Gaussian closed form (`one_cpt_ig` / `two_cpt_ig` with no ODE twin) starts outside its convergence domain at η = 0 — flip-flop regime or (2-cpt) coincident disposition eigenvalues — so it predicts zero (#776); also every uncertainty draw flip-flopped |
//! | `W_IG_FLIP_FLOP` | a twin-carrying inverse-Gaussian absorption model is in the flip-flop regime at typical values (η = 0) for some subject; ferx evaluates the equivalent ODE model there (informational) |
//! | `W_IOV_NO_OCCASIONS` | an IOV model with an Ω_IOV was handed subjects with no occasion label after the occasion rule was applied, so they were simulated at κ = 0 with no inter-occasion variability (#1955, #1980) |
//! | `W_FREM_DRAWN_ETA` | a FREM model was simulated with η drawn from Ω for some subjects (no `match_method`, no supplied η), so their PK follows the drawn covariate η, not their own covariate values (#2245) |
//! | `W_SIMULATED_FLIP_FLOP` | some simulated individuals dosed before an observation drew an η in the flip-flop regime of a twin-less transit / IG closed form, so their dose-driven concentration was clamped to zero (#1823); also counted for the NPDE/NPD reference replicates (#1928) |
//! | `W_UNCERTAINTY_DRAWS_SKIPPED` | some uncertainty draws put a twin-less transit / IG closed form in the flip-flop regime at typical values and were skipped, so intervals built from the rest are biased (#786, #1485) |
//! | `W_RTTE_DEGENERATE` | a simulated subject's recurrent-event hazard is degenerate (expected events non-finite or above 1e6, or the stream passed 1e6 events); its stream was skipped or truncated and censored at the horizon (#762) |
//! | `W_TTE_DEGENERATE_HAZARD` | a simulated subject's analytic hazard draw had a non-positive or non-finite effective rate, so no event was generated and the subject is censored at the observation window (#763) |
//! | `W_STALLED_AT_INIT` | the fit never left its initial estimates (optimizer escape test, or no free θ, Ω or σ moved by 1%), so the reported objective is that of the initial values; silent when `outer_maxiter = 0` |
//! | `W_NO_FEASIBLE_EVAL` | not one outer trial was feasible: every evaluation was rejected by the EBE convergence guard or returned a non-finite objective, so the estimates are the starting values and `converged` is false (#1617) |
//! | `W_ABOVE_START` | a fit reported an objective materially above the one its first evaluation scored — restarts included, which are measured against the fit's start, not their own — typically a start the EBE convergence guard rejected, so it was never a candidate — and `converged` is false (#2461) |
//! | `W_NONFINITE_OBJECTIVE` | the objective at the final estimates is NaN, infinite, or at or above the `1e14` divergence cutoff, so the fit is reported `converged: false` (#1303) |
//! | `W_DEGENERATE_CL_V` | the final estimates put a non-positive or non-finite `CL` or `V` on some dose, observation, EVID=2 or EVID=3/4 record of a 1-/2-/3-cpt IV / oral closed-form model; such a record is served 0 (non-positive / infinite) or `NaN` (#2104) |
//! | `W_EBE_START_DEPENDENT` | the EBEs at the final parameters depend on where the inner loop starts: a cold re-solve scores materially worse or non-finite (#833), lower than the minimised objective (#1565), or the Sobol screen moved subjects (#2013) |
//! | `W_VI_BAD_BASIN` | VI's final ELBO tightness check found the data term more than 25× further above its value at the variational means than expected, so the fit is in a bad basin and reported `converged: false` |
//! | `W_ODE_SOLVER_DIAGNOSTICS` | the ODE solver did not integrate cleanly on that pass: kept min-step clamps, rejected or failed stiff escalations, unfinished, stiff-aborted or diverged segments, or an abandoned non-finite timeline (#1080, #1304) |
//! | `W_ODE_SOLVER_ESCALATION_NOTE` | informational: `ode_method = auto` escalated some segments to a stiff stepper and the integration was otherwise clean (no rejected escalation, no min-step clamp) |
//! | `W_DERIVED_CMT_IOV_UNSUPPORTED` | a `[derived]` expression reads `compartments[i]` on a fit where some subject has IOV (κ); those subjects have no compartment states, so it evaluates to NaN |
//! | `W_DERIVED_CMT_TV_ANALYTICAL` | analytical model with a `[derived]` expression reading `compartments[i]` and subject(s) with time-varying covariates, or individual parameters that read `TIME` (#1855); compartment states are not built for those subjects, so the expression is NaN |
//! | `W_DERIVED_CMT_TV_ODE` | `[odes]` model with a `[derived]` expression reading `compartments[i]` and subject(s) with time-varying covariates, or individual parameters that read `TIME` (#1855); their compartment states come from first-observation parameters and are approximate |
//! | `W_DERIVED_CMT_RESET_ANALYTICAL` | a `[derived]` `compartments[i]` read on an analytical model where some subject has an EVID=3/4 reset; those subjects get NaN states and compartment-based integrals |
//! | `W_DERIVED_CMT_ORAL_DEPOT_INFUSION_ANALYTICAL` | a `[derived]` `compartments[i]` read on an analytical model with a dose superposition cannot place (infusion outside central, bolus outside CMT 1); those subjects get NaN states (#400, #375) |
//! | `W_DERIVED_INIT_ANALYTICAL` | analytical model with an `[initial_conditions]` baseline and a `[derived]` expression reading `compartments[i]`; compartment states are not available, so the expression is NaN (#521) |
//! | `E_RUV_MAGNITUDE_NONPOSITIVE` | a custom residual-error magnitude (e.g. a `weight = ...` modifier) or a `power(SIGMA, P)` exponent evaluates to zero, negative or non-finite at an observation, at the model's initial θ (#484, #1182) |
//! | `E_DOSE_ATTR_DOUBLE_USE` | a dose attribute the engine applies at the dose event (`F`, `LAGTIME`/`ALAG`, `F{n}`/`ALAG{n}`, or a `D{n}`/`R{n}` a coded `RATE` lands on) is also read by the model, so it is applied twice (#993, #1004) |
//! | `E_NEGATIVE_LAGTIME` | a dose lag (`lagtime` / `ALAG{n}`, or a per-route `lag=` on the dosed compartment) is negative at the fit's initial typical values at some dose record's own snapshot; NONMEM refuses the same run (#1189) |
//! | `E_DOSE_ATTR_NONFINITE` | a dose attribute (lag time, bioavailability, or a modeled `D{n}`/`R{n}` behind a coded `RATE`) is `NaN` or infinite at typical values at some dose record's own snapshot (#1189, #1235) |
//! | `E_ETA_NOT_DECLARED` | an unresolved identifier reads as a random effect (`ETA…` / `KAPPA…`) but no `omega` / `kappa` declaration defines it and the data has no such column |
//! | `W_ETA_NOT_DECLARED` | as `E_ETA_NOT_DECLARED`, reported without `--data`, where the dataset might still supply the column |
//! | `E_IOV_MISSING_OCC` | the model declares kappa (IOV) parameters but no occasion labels were found (no `iov_column` read and no `iov_occasion` derivation) |
//! | `E_IOV_SINGLE_OCCASION` | a derived `iov_occasion` rule (`dose` / `time(...)`) gives every subject at most one occasion, so the per-occasion κ is confounded with the η and unidentifiable (#757). Block `fit_options`. Raised by `fit()` on the labelled population and by `run_model_simulate` on its `[simulation]` template before simulating (#2052), each with its own advice (#1788), and by `ferx check --data`, which derives the model's rule on the data the same way |
//! | `E_NONFINITE_DV` | a subject carries a non-finite (`NaN`/`Inf`) observation, usually a simulation-reader population passed to `fit()` |
//! | `W_DESIGN_DV` | simulation reader only: `EVID=0` rows with a missing `DV` were kept as design points to simulate at (#957) |
//! | `W_INIT_NOT_REPRESENTABLE` | a declared initial value the packer itself moved before any box existed — the `1e-10` log floor, or the Fisher-z `±3` rail on a free `block_sigma` correlation (#1307) |

use serde::Serialize;

/// Severity of a [`Diagnostic`]. Only `Error` affects the `ferx check` exit
/// code and is treated as fatal by [`first_error`].
///
/// `Info` is a row that reports a fact about the model or data rather than a
/// finding — `W_LEVEL_SUMMARY`'s per-level counts (#2387). `ferx check` prints it,
/// but it is not counted as a warning ([`CheckReport::warning_count`]), so a model
/// whose only rows are informational validates clean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// A single validation finding.
///
/// `line` is **block-level** in the current implementation: it points at the
/// `[block]` header the finding belongs to, not the exact offending token.
/// Token/column spans are a deferred enhancement (see the plan and the
/// check-report docs).
#[derive(Debug, Clone, Serialize)]
pub struct Diagnostic {
    pub severity: Severity,
    /// Stable machine-readable code, e.g. `"E_MISSING_COVARIATE"`.
    pub code: String,
    /// Human-readable description (the historical free-text message).
    pub message: String,
    /// Owning block, e.g. `"individual_parameters"`, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block: Option<String>,
    /// 1-based line of the owning block's header, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
    /// Actionable hint, e.g. `"available covariates: WGT, AGE, SEX"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

impl Diagnostic {
    /// An `Error`-severity diagnostic with the given code and message.
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Error,
            code: code.into(),
            message: message.into(),
            block: None,
            line: None,
            suggestion: None,
        }
    }

    /// A `Warning`-severity diagnostic with the given code and message.
    pub fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Warning,
            code: code.into(),
            message: message.into(),
            block: None,
            line: None,
            suggestion: None,
        }
    }

    /// An `Info`-severity diagnostic with the given code and message: a fact, not a
    /// finding, which no validation consumer counts as a warning (#2387).
    pub fn info(code: impl Into<String>, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Info,
            code: code.into(),
            message: message.into(),
            block: None,
            line: None,
            suggestion: None,
        }
    }

    /// Attach the owning block name (builder style).
    pub fn with_block(mut self, block: impl Into<String>) -> Self {
        self.block = Some(block.into());
        self
    }

    /// Attach the owning block's header line (builder style).
    pub fn with_line(mut self, line: usize) -> Self {
        self.line = Some(line);
        self
    }

    /// Attach an actionable suggestion (builder style).
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// True for `Error`-severity diagnostics.
    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }

    /// True for `Warning`-severity diagnostics.
    pub fn is_warning(&self) -> bool {
        self.severity == Severity::Warning
    }
}

/// The full result of a `ferx check` run.
#[derive(Debug, Clone, Serialize)]
pub struct CheckReport {
    /// True when no `Error`-severity diagnostics are present.
    pub valid: bool,
    /// Model name / file stem.
    pub model: String,
    /// Data file path, when `--data` was supplied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    pub diagnostics: Vec<Diagnostic>,
    /// The `[individual_parameters]` block as the `[covariate_model]` desugar
    /// rewrote it (#1111), one line per assignment. Empty for a model that
    /// declares no such block.
    ///
    /// `ferx check` prints this so the expression the block actually built is
    /// visible — a covariate model stated declaratively is otherwise
    /// unauditable against NONMEM, since nothing in the file spells out the
    /// centring constant, the missing-value guard or where the factor landed.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub desugared_individual_parameters: Vec<String>,
}

impl CheckReport {
    /// Build a report from collected diagnostics; `valid` is derived as
    /// "no error-severity diagnostics present".
    pub fn new(
        model: impl Into<String>,
        data: Option<String>,
        diagnostics: Vec<Diagnostic>,
    ) -> Self {
        let valid = !diagnostics.iter().any(Diagnostic::is_error);
        CheckReport {
            valid,
            model: model.into(),
            data,
            diagnostics,
            desugared_individual_parameters: Vec::new(),
        }
    }

    /// Count of `Error`-severity diagnostics.
    pub fn error_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| d.is_error()).count()
    }

    /// Count of `Warning`-severity diagnostics; `Info` rows are not warnings.
    pub fn warning_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| d.is_warning()).count()
    }

    /// Count of `Info`-severity diagnostics (#2387).
    pub fn info_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Info)
            .count()
    }
}

/// Collapse a slice of diagnostics to a fail-fast `Result`: `Err` carrying the
/// first error-severity [`Diagnostic`], else `Ok`. This lets the entry points
/// keep their fail-fast behavior and identical error strings (the `Display` of
/// the returned [`EngineError`] is the diagnostic's message) while sharing the
/// diagnostic-producing validators with `ferx check` — and, since #1746, while
/// handing the caller the same code, block and suggestion `ferx check` reports.
pub fn first_error(diagnostics: &[Diagnostic]) -> Result<(), EngineError> {
    match diagnostics.iter().find(|d| d.is_error()) {
        Some(d) => Err(EngineError::from_diagnostic(d.clone())),
        None => Ok(()),
    }
}

/// The error an entry point (`predict*`, `simulate*`, `compute_npde_npd`,
/// [`crate::run_sir`], [`crate::run_covariance`], `inits_from_nca` since #1746; [`crate::fit`]
/// and the file entry points built on it — [`crate::fit_from_files`],
/// [`crate::prepare_run`], [`crate::run_model_with_data`], [`crate::run_model_simulate`]
/// and their variants — since #1772) returns when it refuses its input or fails.
///
/// When `ferx check` reports the same refusal with a code, the error carries that
/// [`Diagnostic`] — so a caller matches on [`code`](Self::code) instead of parsing
/// the message. Some refusals have a code although `ferx check` cannot reach
/// them, since it has no parameter vector, no fit, no fit's bindings, no simulation
/// options and no adaptive controller: `E_THETA_LENGTH`, `E_PARAM_SHAPE`, the
/// binders' `E_THETA_LEVEL_BINDING` / `E_COVARIATE_STATS_BINDING` on a fit's
/// bindings, `E_POPULATION_MISMATCH`, `E_FIT_SUBJECTS_MISMATCH`, the post-hoc
/// steps' `E_FIT_INPUT_MISSING` / `E_FIT_INPUT_CHANGED` / `E_FIT_INPUT_MODEL` /
/// `E_SIR_OPTIONS` / `E_SIR_NOTHING_TO_SAMPLE` / `E_ODE_SOLVER_OPTIONS`, the
/// uncertainty draws' `E_LOGIT_THETA_OUT_OF_RANGE` (#2044), and the
/// predict / simulate / adaptive codes of #1774 (`E_CTMM_UNSUPPORTED`,
/// `E_TTE_NOT_SIMULATABLE`, `E_SIMULATE_HORIZON`, `E_PROPENSITY_INPUT`,
/// `E_ADAPTIVE_UNSUPPORTED`, `E_ADAPTIVE_OPTIONS`, `E_ADAPTIVE_SPEC`,
/// `E_ADAPTIVE_CONTROLLER_ACTION`), `E_SUPPLIED_ETAS` (#1594), and the file entry
/// points' `E_DATA_PATH_MISSING`, `E_SIMULATION_DESIGN` and `E_IOV_SINGLE_OCCASION`
/// (#1788).
/// A failure with no code carries none ([`diagnostic`](Self::diagnostic) is `None`).
///
/// `Display` is the historical `String` error byte for byte: the message, preceded
/// by `"{context}: "` when the refusal names its entry point (`run_sir`,
/// `run_covariance`). `.to_string()` therefore recovers the pre-#1746 `Err`.
///
/// Four refusals also fold their suggestion in after the message: `predict`'s
/// unbound `theta NAME[...]` block, `E_POPULATION_MISMATCH`,
/// `E_FIT_SUBJECTS_MISMATCH` and `E_LOGIT_THETA_OUT_OF_RANGE`. There `Display`
/// appends it, while
/// [`message`](Self::message) stays the bare message and the suggestion is only in
/// [`suggestion`](Self::suggestion). So a renderer shows either `to_string()` alone,
/// or `message()` and `suggestion()` side by side — never the advice twice.
#[derive(Debug, Clone)]
pub struct EngineError {
    diagnostic: Option<Diagnostic>,
    message: String,
    context: Option<String>,
    /// `Display` appends the diagnostic's suggestion as a sentence after the message.
    suggestion_in_display: bool,
}

impl EngineError {
    /// An error carrying `d`, whose message is `d.message`.
    pub(crate) fn from_diagnostic(d: Diagnostic) -> Self {
        EngineError {
            message: d.message.clone(),
            diagnostic: Some(d),
            context: None,
            suggestion_in_display: false,
        }
    }

    /// An error carrying `d` for its code, block and line, whose message is `message`
    /// rather than `d.message`: for a refusal whose historical text the diagnostic
    /// rewords (`ferx check` renders `block:line` beside a parse error's message, so its
    /// diagnostic drops the ` (line N)` the raw text carries; #2062 review B1).
    pub(crate) fn from_diagnostic_keeping_text(d: Diagnostic, message: String) -> Self {
        EngineError {
            message,
            ..EngineError::from_diagnostic(d)
        }
    }

    /// An error carrying `d` whose `Display` folds the suggestion in after the
    /// message, capitalised and closed with a full stop — for a refusal whose
    /// historical text read that way. [`message`](Self::message) stays `d.message`.
    pub(crate) fn with_suggestion_in_display(d: Diagnostic) -> Self {
        EngineError {
            suggestion_in_display: true,
            ..EngineError::from_diagnostic(d)
        }
    }

    /// An error with no diagnostic: [`code`](Self::code) is `None`.
    ///
    /// Right only for a failure `ferx check` has no code for: a runtime, IO or
    /// internal failure, or a refusal no code exists for yet. A refusal some
    /// helper already codes goes through that helper's [`Diagnostic`] instead.
    /// There is deliberately no `From<String>` (#1775): a `?` on a `String`
    /// helper would otherwise drop a code silently. Every call site is listed in
    /// `UNCODED_SITES` (the test module below) with its reason.
    pub(crate) fn uncoded(message: impl Into<String>) -> Self {
        EngineError {
            diagnostic: None,
            message: message.into(),
            context: None,
            suggestion_in_display: false,
        }
    }

    /// This error, attributed to the entry point `context` (e.g. `"run_sir"`).
    pub(crate) fn in_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// The diagnostic `ferx check` reports for this refusal, if it has one.
    pub fn diagnostic(&self) -> Option<&Diagnostic> {
        self.diagnostic.as_ref()
    }

    /// The diagnostic's stable code (e.g. `"E_COV_LEVEL_UNKNOWN"`), if any.
    pub fn code(&self) -> Option<&str> {
        self.diagnostic.as_ref().map(|d| d.code.as_str())
    }

    /// The model block the diagnostic belongs to, if known.
    pub fn block(&self) -> Option<&str> {
        self.diagnostic.as_ref().and_then(|d| d.block.as_deref())
    }

    /// The diagnostic's actionable hint, if any.
    pub fn suggestion(&self) -> Option<&str> {
        self.diagnostic
            .as_ref()
            .and_then(|d| d.suggestion.as_deref())
    }

    /// The message, without the entry-point context and without the suggestion.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The entry point the refusal names, e.g. `Some("run_sir")`.
    pub fn context(&self) -> Option<&str> {
        self.context.as_deref()
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(c) = &self.context {
            write!(f, "{c}: ")?;
        }
        f.write_str(&self.message)?;
        if self.suggestion_in_display {
            if let Some(s) = self.suggestion() {
                let mut chars = s.chars();
                if let Some(first) = chars.next() {
                    write!(f, " {}{}.", first.to_uppercase(), chars.as_str())?;
                }
            }
        }
        Ok(())
    }
}

impl std::error::Error for EngineError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_set_fields() {
        let d = Diagnostic::error("E_MISSING_COVARIATE", "covariate 'WT' not found")
            .with_block("individual_parameters")
            .with_line(11)
            .with_suggestion("available: WGT, AGE");
        assert!(d.is_error());
        assert_eq!(d.code, "E_MISSING_COVARIATE");
        assert_eq!(d.block.as_deref(), Some("individual_parameters"));
        assert_eq!(d.line, Some(11));
        assert_eq!(d.suggestion.as_deref(), Some("available: WGT, AGE"));
    }

    #[test]
    fn report_validity_derives_from_errors() {
        let ok = CheckReport::new("m", None, vec![Diagnostic::warning("W_X", "heads up")]);
        assert!(ok.valid);
        assert_eq!(ok.error_count(), 0);
        assert_eq!(ok.warning_count(), 1);

        let bad = CheckReport::new("m", None, vec![Diagnostic::error("E_X", "nope")]);
        assert!(!bad.valid);
        assert_eq!(bad.error_count(), 1);
    }

    /// #2387: an `Info` row keeps the report valid and is counted as neither an error
    /// nor a warning, and it serialises as `"info"`. Mutations — `warning_count` back to
    /// `!is_error()` → the warning count reads 2; `info_count` filtering on anything
    /// else → it reads 0 or 2.
    #[test]
    fn info_rows_are_not_warnings() {
        let report = CheckReport::new(
            "m",
            None,
            vec![
                Diagnostic::info("W_LEVEL_SUMMARY", "level `ARM`: 7 groups"),
                Diagnostic::warning("W_X", "heads up"),
            ],
        );
        assert!(report.valid);
        assert_eq!(report.error_count(), 0);
        assert_eq!(report.warning_count(), 1);
        assert_eq!(report.info_count(), 1);
        assert!(first_error(&report.diagnostics).is_ok());
        let d = &report.diagnostics[0];
        assert!(!d.is_error() && !d.is_warning());
        let json = serde_json::to_value(d).unwrap();
        assert_eq!(json["severity"], "info");
    }

    #[test]
    fn first_error_returns_first_error_message() {
        let diags = vec![
            Diagnostic::warning("W_A", "warn first"),
            Diagnostic::error("E_B", "second is the error"),
            Diagnostic::error("E_C", "third"),
        ];
        let err = first_error(&diags).expect_err("an error-severity diagnostic");
        assert_eq!(err.to_string(), "second is the error");
    }

    /// #1746: the `Err` carries the whole diagnostic `ferx check` reports — code,
    /// block, line and suggestion — and its `Display` is the message alone, as the
    /// `String` it replaced was. A failure with no diagnostic carries no code.
    ///
    /// Mutations: build the error from `d.message` only (a `From<String>`) → the
    /// `code()` assert dies; drop `suggestion` / `block` in the copy → that assert
    /// dies; prefix `Display` with the code → the `to_string()` asserts die.
    #[test]
    fn first_error_carries_the_diagnostic_and_displays_its_message() {
        let mut coded = Diagnostic::error("E_B", "second is the error")
            .with_block("covariate_model")
            .with_suggestion("bind it first");
        coded.line = Some(7);
        let diags = vec![Diagnostic::warning("W_A", "warn first"), coded];
        let err = first_error(&diags).expect_err("an error-severity diagnostic");
        assert_eq!(err.code(), Some("E_B"));
        assert_eq!(err.block(), Some("covariate_model"));
        assert_eq!(err.suggestion(), Some("bind it first"));
        assert_eq!(err.diagnostic().and_then(|d| d.line), Some(7));
        assert_eq!(err.message(), "second is the error");
        assert_eq!(err.context(), None);
        assert_eq!(err.to_string(), "second is the error");

        let plain = EngineError::uncoded("no diagnostic here");
        assert!(plain.diagnostic().is_none());
        assert_eq!(plain.code(), None);
        assert_eq!(plain.to_string(), "no diagnostic here");

        // Context is printed ahead of the message and kept out of `message()`.
        let attributed = plain.in_context("run_sir");
        assert_eq!(attributed.context(), Some("run_sir"));
        assert_eq!(attributed.message(), "no diagnostic here");
        assert_eq!(attributed.to_string(), "run_sir: no diagnostic here");

        // A folded refusal prints the advice once, after the message, while `message()`
        // and `suggestion()` keep them apart (review r1 #6). Mutations: fold into
        // `message` instead → the `message()` assert dies; drop the fold from `Display`
        // → the `to_string()` assert dies; skip the capital → it dies too.
        let folded = EngineError::with_suggestion_in_display(
            Diagnostic::error("E_X", "Short.").with_suggestion("with the advice folded in"),
        )
        .in_context("predict");
        assert_eq!(folded.code(), Some("E_X"));
        assert_eq!(folded.message(), "Short.");
        assert_eq!(folded.suggestion(), Some("with the advice folded in"));
        assert_eq!(
            folded.to_string(),
            "predict: Short. With the advice folded in."
        );
        // The fold is the exception: an ordinary diagnostic with a suggestion prints
        // its message alone.
        let plain_coded = EngineError::from_diagnostic(
            Diagnostic::error("E_X", "Short.").with_suggestion("advice"),
        );
        assert_eq!(plain_coded.to_string(), "Short.");
    }

    #[test]
    fn first_error_ok_when_no_errors() {
        let diags = vec![Diagnostic::warning("W_A", "just a warning")];
        assert!(first_error(&diags).is_ok());
    }

    #[test]
    fn optional_fields_omitted_in_json() {
        let d = Diagnostic::error("E_PARSE", "bad");
        let json = serde_json::to_string(&d).unwrap();
        // block / line / suggestion are None → skipped.
        assert_eq!(
            json,
            r#"{"severity":"error","code":"E_PARSE","message":"bad"}"#
        );
    }
}

#[cfg(test)]
#[path = "diagnostics_uncoded_sites.rs"]
mod uncoded_sites;

#[cfg(test)]
#[path = "diagnostics_code_registry_tests.rs"]
mod code_registry;
