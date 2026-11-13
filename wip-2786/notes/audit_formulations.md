# #2786 minimal-formulation audit (2026-11-12), per consumer — what to implement

1. Laplace/AGQ grad (run_agq ode_adjoint.rs:5792, agq_taped :6100): today generic taped record per node (DualV, all lanes) + 2nd record at η̂ + 2 vsweep_h. Minimal: L2 forward at η̂ (= p1), reverse#1 seeded adj(A) on f,G,Q lanes → ∇ηA total (3rd order contraction free) + ∂θA; solve w; reverse#2 at L1 seeds [wᵀ∇l, rows]. Nodes: F64 forward + F64 reverse each; u_bar → b-adjoint fed through scalar tape. No eigen trick, no node lanes.
2. Covariance R (cov_hessian hessian.rs:248, mode_response :428, pass :922): mode_response duplicates run()'s ModeOut (covariance.rs:1345 calls grad then diag Hessian at same point). K=4 chunking is tape-memory legacy: do one cross forward with all n_axes free dirs. Still generic: FOCE f(η=0) :1082, AGQ nodes :1160, ng_out DualH<8,36>, IOV gate :1012, exact anchor DualT<4,DualH>. Exact anchor/NG/AGQ → tangent(reverse(L2 forward)) = forward-over-reverse (rhs_ir::tangent, written).
3. NG terms: full 3rd tensor via M seeds but only contracted (objective_ng_eta_dual :5712, ng_eta_contraction :4685) → 1 seed. ng_theta_adjoint_seeds planned path does one call per seed + extra forward (terms:399) → multi-seed. REV_BUF single slot gets clobbered by Gaussian reverse → explicit forward-state handle. Value-only node parts at layout2 → F64. Duplicate plans: inner loop per-endpoint TermKind plans vs outer combined NonGaussian plan → one combined plan. DTMM missing from non_gaussian_eta_grad (verify).
4. FOCE/FOCEI grad (run :4981): eta_derivatives :5636 uses DualT<N,f64>; L1 reverse recomputes an L1 forward that's a subset of the L2 forward → reverse reads L2 registers strided (forward-state handle). ChannelPlan mu-scale & covariate_nn still generic DualV (:5070). kjit_verify_taped :2774 = prototype, delete.
5. GN/BHHH/S-matrix/trust-region fixed-EBE score (fixed_ebe_score :539, gauss_newton.rs:2322): two passes at same point; g_i is one more seed of run's reverse.
6. IOV: dense payload over stacked [η,κ] with structurally-zero κj-κj' blocks → per-occasion live width / sparse pair list; fixed_ebe_score_iov :690, record_ode_iov :1682 generic.
7. Gradient-only η consumers (inner BFGS exact_obs_grads_into :4018, exact_inner_gradient :4285, HMC, SAEM-IMH, VI :3981): L1 forward → F64 forward + 1-seed reverse (N≥3). SAEM M-step θ score fixed_eta_gradient.rs:631/734/3886 generic DualV<1,0> → F64 fwd + 1-seed rev.
8. Mixtures fine; mixing_logp_hess_exact model_parser.rs:19474 DualT<8,R> small → port.
Remaining dual users: focei_kernel.rs (:145, :502 focei_rows DualT), eta_jets.rs DualF, survival eta_jets2_by_bucket/eta_gradient_by_bucket (fallback+oracle), kernel_jit.rs:521 consts borrowed from DualF, model_parser.rs:33639 Dual2<0> f64 stand-in, residual_error.rs:1250 residual_rd2_corr Dual2<1>; strings in saem.rs/inner_optimizer.rs/validation.rs/build_info.rs. Legit local: special.rs gamma_pq_partial Taylor, categorical Mills Taylor.
Cross-cutting plan API: forward-state handle (reverse never recomputes forward), strided lower-layout reverse, tangent of reverse (rhs_ir::tangent), sparse/per-occasion payloads.

## Status (2026-11-12, session 2)
- DONE (built, regress pending): Fwd handle (Plan::forward_kept/forward_layout→Option<Fwd>/reverse_from with prefix compaction incl. loop logs; REV_BUF/reverse_seeds removed). run(): L1 θ-reverse + FOCE pop F64 reverse read kept forwards.
- DONE: run() steps 2+3 merged: one R recording of (F, ∇l), (1+N)-lane sweep; ∇ηF/H via lane chain (G, Q) + NG contraction; Ψ/rows = combine_lanes coefs. eta_derivatives + objective_ng_eta_dual (DualT) deleted.
- DONE: NgJets.t → NgThird {Full, Planned(plan,fwd)}; contract_t = 1 reverse (was N(N+1)/2 seeds).
- DONE: Padé squarings counted as data (select chain, RECORDED_SQUARINGS=8): exact, ctmm bad 0 guard 0. note_inexact hook removed.
- DONE: rhs_ir::tangent (forward-mode of any Func; forward-over-reverse) + test forward_over_reverse_is_the_hessian_on_random_programs (not yet run); Pentagamma shim; rhs_jit shim table dedup (Shim1::f).
- WRITTEN, untested: src/sens/ode_adjoint_agq.rs agq_planned (AGQ/Laplace via plan: 2 reverses at η̂ + per-node L1 fwd/F64 rev), wired before taped fallback in run_agq.
- TEMP scaffolding to strip: kernel_jit GLOG/guard_log_diff/br_value pin print, calls in terms.rs + ode_adjoint.rs; kernel_jit_loop debug prints.
- Scripts: kj/vsmain.sh (vs merge-base ref /scratch/mu/ad/ferx-main-feat from worktree /scratch/mu/ref-main), kj/vsagq.sh (agqex variants).
- NEXT: covariance via tangent(reverse) (exact anchor, NG, AGQ nodes), cov mode_response dup, K=4 chunking; GN/S-matrix score as extra seed; gradient-only η consumers → F64 fwd + 1-seed rev; SAEM M-step score; IOV sparse; delete duals.

## Covariance via tangent (design, 2026-11-12)
R-matrix columns = HVPs of Ψ_w along u_a=(e_a, η′e_a). Planned gradient pipeline: plan L2 fwd (f,G,Q) → R-tape kernel Ψ → adj (af,aG,aQ) → plan reverse. HVP = tangent of that pipeline.
- FOCEI/FOCE (no Q): existing PlannedFg = forward at cross layout + rblock with plan reverse at cross layout = reverse-over-forward(L1 ⊗ free). OK already.
- Exact anchor / Laplace / NG (Q or ∇²q enters Ψ): needs Q's u-tangents = third-order mixed (free×η×η). Do NOT add a nested layout: compile chunk kernels as tangent(forward(L2), K) and reverse(tangent(forward(L2),K)) via rhs_ir::tangent (Func→Func). Key (layout, rev, tan K) in ChunkProgram::get. Tangent args: arg n+a, offset off*K+j → per-slot tangent block stride l*K ([lane][dir]). Lowering (lower_chunk_opt / slot_addr) needs the tangent arg's stride; lane tier can't run Funcs → tangent kernels native-only (compile at first use; covariance is once per fit, chunks shared).
- Plan API: forward_tan(layout, K, u, p, du_tan) -> Fwd(+tangent buf) ; reverse_tan_from(...) for block reverse. Loop calls: tangent of loop bodies (LoopKernels per (layout,rev,tan)) + run_loop with tangent buffers; run_loop_rev likewise.
- Then hessian.rs: PlannedFg generalizes to layout L2 with tangent K (exact anchor, ng_out → term plans with tangent), AGQ nodes: L1 ⊗ tangent forward per node + block.
- Also: mode_response duplicates run()'s ModeOut (covariance.rs:1345) → reuse; K=4 chunking was tape-memory driven → single pass with all directions when plans serve.

## Perf status (2026-11-12, after tangent covariance)
- Large A/B vs main (best of 2): pk2 0.92, wiov 0.70, pk2_ode 1.19, mm 1.08 (OFVs equal). Gate: all ≤1.
- DONE: tangent kernels start on interpret_func (scalar Func interpreter) → three_cpt 2.2→1.4 s (main 0.76).
- DONE: spectral eigenvalues in block (graph) order, not value order → kills `exact(k)` flips.
- DONE: Layout.free + cross() removed (tangent layouts replace them); FREE bit kept as "depends on input".
- OPEN pk2_ode: record 944 ms + compile 363 ms of 4.4 s; guard-fails 368; remaining churn = spectral `ok` (linear.rs:647 br_value) flipping spectral↔Padé per segment (regime boundary ka≈α), combos per segment exceed 4 variants. Options: (a) derivative types always Padé (f64 keeps spectral; rounding-level mismatch, plan-vs-dual validation still consistent); (b) record both arms + select with live-masked reverse (needs dead-arm masking in Flat/rhs_ir to avoid 0·inf); (c) variant per segment. Then measure steady-state per-iteration cost vs main.
- OPEN mm (nonlinear ODE, RK loop) 1.08: not yet profiled.
- MEASURED+REVERTED: spectral record margin (8×): pk2_ode 5.47 s vs 4.4 s; Padé-only recordings: 61 s, 3206 guard fails (Padé churns on LU pivot pins linsolve.rs:203 br_value and squaring bound linear.rs:291). Spectral `ok` guard kept as before.
- NEXT (compile/record optimality, user hint "ensure compilation is truly optimal"): perf where record 944 ms goes (Recording::key formats code as Debug string kernel_jit.rs:~1324; chunk key format!("{:?}") ~2291 → hash Flat directly); in-kernel LU pivoting (pivots as T through rosenbrock/linear/solve_linear_system_g; drop HostFn::LuPivots); then sticky Padé on spectral-ok flip; tier policy: promotion synchronous, up to 3 compiles per kernel → consider background compile thread / skip tiers.
- MATH LEAD (linear segments, kill spectral `ok` churn): exp(Aτ) = p_λ(A), p_λ the Newton interpolant of z↦e^{zτ} at λ = spec(A). ∂p_λ(A)/∂λ_j = f[λ,λ_j]·χ_λ(A) = 0 by Cayley–Hamilton ⇒ λ's derivatives enter only at 2nd order (via ∂χ/∂A), and there with ε/gap amplification only (acceptable to gap ~1e-4). Divided differences stable near coalescence via exprel family (f[a,b]=e^{bτ}τ·exprel((a−b)τ)); 3+ nodes need confluent-stable DDs (Opitz: exp of bidiagonal, or Taylor near-confluent, McCurdy–Ng–Parlett). Would remove the projector/κ/gap regime test except complex eigenvalues. RISK: Newton form stability on stiff spectra (cancellation in Π(A−λ_i)), node ordering must be structural (block order). Prototype + measure vs Padé/spectral before adopting.

## Re-audit of DONE work from the math (user: rule applies to past + future work)
- FOCE/FOCEI grad: L2 forward kept — alternative L1 fwd + (1+N)-seed L1 rev costs (1+N)² lane-units ≥ L2's 1+N+N(N+1)/2 for N≥3. OK.
- AGQ nodes: only Gᵀ·adj needed, never G → F64 forward per node as rblock(ins=b, outs=f_node) whose reverse = plan F64 reverse (u_bar = b-adjoint) and accumulates p_bar (θ) per lane into a side buffer. Saves N lanes/node. TODO.
- Covariance: K=4 chunks were dual-tape memory; tangent layout with K = all directions in ONE pass (base forward once). mode_response dup of run()'s ModeOut. TODO.
- LU pivoting in-kernel (select partial pivoting, pivots as T through factor/solve; drop HostFn::LuPivots + br_value pins). TODO.
- Linear segments: Newton/divided-difference exp (see MATH LEAD) to remove spectral `ok` regime switch. PROTOTYPE.
- Compile/record: DONE structural Flat Hash + bitwise `same`, chunk cache by structure, Fx hasher for recorder maps (perf showed ~6% Debug fmt + ~6% SipHash). Next: tier policy (sync promotion, up to 3 compiles).
- USER (2026-11-12): ODE solvers may be rewritten to work better with optimal gradients.
- DESIGN LEAD: dense linear solve as a differentiable IR primitive (host call with lane rules): value LU once at runtime (pivoting = runtime branch, no guards/selects), lanes as extra RHS: ẋ=A⁻¹(ḃ−Ȧx); x_kl=A⁻¹(b_kl−A_kl x−A_k x_l−A_l x_k); reverse b̄=A⁻ᵀx̄, Ā=−b̄xᵀ. O(n³+n²·lanes) vs O(n³·lanes). Applies to Rosenbrock W, Padé denominator, SS solves. Needs rhs_ir Host ops with forward lane rules + a reverse host rule.
- A/B after in-kernel LU: pk2_ode 1.19→1.09, mm 1.08→1.17 (mm regressed).

## SS fixed-point call (design, 2026-11-12)
Plan Call::Fixed { sub: sub-plan of one cycle P (chunks + ODE loop calls), inputs θ=[u (n), caps], n_eta=0 (dense param lanes, no one-hot folding) }.
- Forward @L: value: same Anderson (+linear closed-form check) as f64 path, P via sub @F64 → u*_val (bitwise = f64 path; f64 path drops the FD Newton correction — values only). J = ∂P/∂u: sub @Layout{n} with u lanes one-hot, caps lanes 0 → M=(I−J)⁻¹ (value, constant). Lanes: K=2 steps u←u+M(P(u)−u) with sub @L (u lanes, caps' outer lanes), u_0=u*_val with zero derivative lanes; output value lane := u*_val (exact), derivative lanes from step K (exact to order 2: dΦ/du=I−M(I−J)=0).
- Reverse @L: reverse of the K steps with kept Fwds: w=Mᵀū; ū←ū−w+P_uᵀw; caps̄+=P_capsᵀw; ū_0 dropped.
- Recording: cut P's code out of the open recording like record_body (captures = outer reads) but as a sub-Recording → Recording::plan (nested loop calls allowed). Replaces: dual SS cycles with .val() stop (ode_provider 2012–2074), Anderson in recordings, FD-Jacobian newton correction (FD gone).

## Status 2026-11-12 late
- DONE: host-LU in recordings (lu_factor_recorded/lu_solve_recorded: HostFn::LuFactor + 2 refinement steps, value-preserving) → mm 1.17→1.02; select-LU removed. Tests: plan_tests (linear solve exact to 2nd order across pivot change; fixed point implicit derivs) PASS.
- DONE: SS fixed-point call (kernel_jit_fixed.rs, CallKind/CallLog, record_fixed, ss_solve_g, PkNum::ss_fixed_point; sub-plans dense params). mm has no SS doses (earlier attribution wrong) — needs a model with SS for coverage.
- DONE: AGQ nodes = F64 forward + tape block (node_block) — verifying.
- MEASURED: Padé-only recordings 60 s (compile 10 s) → spectral stays.
- pk2_ode: coalescence flips are per subject (generator shared by segments) → ≤1 extra record/subject; remaining gap = first record+compile (~1 s of 3.5). NEXT: compile tiers (each kernel compiled up to 3×), record-path cost.
- TODO list: cov single pass (needs kernel-level replacement of DualT<K,R>); mode_response dup; loop runtime log arena; lane-aware host primitive (optional); delete duals; cleanup scaffolding (GLOG, debug prints, FERX_KJIT flag); docs/changelog/API baseline/PR.
- DONE: background compile thread (compile_queue; Tiered.want/queued/me; compile once at wanted tier). A/B: pk2 0.70, pk2_ode 0.91, wiov 0.72, mm 1.01 (best of 3; noisy).
- DONE: CallKind::Repeat (RepeatSpec/record_repeat, PkNum::repeat); Padé squarings via repeat, scale from PadeSquarings host (2nd output) — no headroom/selects. mctmm 1591→427 ms (main 173). Fixed re-entrancy: reverse_from takes ADJ out of TLS (sub-plan reverses nest).
- Tests in src/parser/kernel_jit_plan_tests.rs: solve (2nd order, pivot change), fixed point (implicit 2nd order), expm (squaring count varies), every call kind reverse vs forward.
- NEXT: verify ctmm runs; vsmain; then dual deletion program (counts code-only refs: DualV 160, DualF 45, DualT 85, DualH 49, Dual1 48, Dual2 341, DualMixed 96, Taylor 22) — migrate consumers first (inner η-grad/H, HMC, SAEM, Bayes, IOV, nn, channels, focei_kernel DualT, covariance pass), then delete types & port tests.
- DONE: plans always on (FERX_KJIT & other env switches removed except FERX_KJIT_DEBUG); compile pool (one thread per core, queue-driven).
- Full lib suite w/ plans on: 273 "failures" mostly contamination (abort from panic in NLopt callback + poisoned lock). Real: impmap test (reset cache between runs — fixed), nonfinite poisoned() helper (clear indiv_param_program — fixed), NaN guard in lu_solve_recorded value-preserve (select), bit-identity tests (fit_inputs ltbs/pre_logged, covariance stationarity/mixture, IS mixture) — cause: dual-comparison scaffolding mixing generic/plan paths.
- DONE (stage 1): removed kjit_check_layout, kjit_verify_taped/KJIT_TAPED_*, spot checks (kjit_jets1, kjit_values), PlannedFg verdict/settle, term-plan 3-point validation. Remaining dead: Plan.checked/validated/values_verdict/taped_* fields + layout_verdicts in kernel_jit.
- OPEN: plan-cache soundness across objects (address reuse) — content fingerprint or call-scope; Drop impl rejected (breaks struct-update API).
- Plan cache key now: model addr + model_fingerprint (structure) + subject content fingerprint + mixture class + IOV layout fingerprint + n/n_theta/kind. Fixes: mixture IS (class), IOV (layout), cross-test reuse (fingerprints). Generations/Drop approach removed.
- Removed kjit_values (plan-served likelihood values): determinism (history-dependent ULP differences). Values always from f64 predictor.
- Removed Plan scaffolding fields (checked/runs/values_*/taped_*), layout_verdicts, SPOT_EVERY.
- Fixed: record_body / record_sub guard register shared by CSE with an outer register → treated as capture (panic "no entry found for key").
- TEMP debug in kjit_jets1 (plan vs generic jets print) — remove with duals.

## Order gates (2026-11-12, session 3)
- ROOT CAUSE (CTMM outer grad 1.4e-2, NG cov Hessian 3.2e-2): host-value factorization (LU) / value Jacobian (Fixed M, spectral J0) is StopGrad data; K refinement/Newton/chord steps make lanes exact to order K, but a reverse holds the factor fixed so its adjoint is exact only to order K-1 of the forward lanes (error enters at order >= K in the combined η,θ jet). θ-reverse of second-order η lanes = 3rd order -> needs K=3; FoR at L2+tan -> K=4.
- FIX: Flat::Gate{d,a,b,k} (PkNum::order_gate) resolved per (layout, rev) in ChunkProgram::get (resolve_gates: alias + Flat DCE rooted at Deriv/Host) with g = layout.order() + rev (Layout::order = 2|1|0 + tan>0). LU refine steps 1..=4 gated, spectral chord steps gated, Fixed n_steps = order + log.is_some() at run time (also fixes F64 reverse through Fixed, previously no log). Value forwards now run 0 refinement steps.
- Test: kernel_jit_plan_tests every_call_kind_reverses_exactly_from_second_order_lanes (reverse from h lanes vs forward tangent + ∂³ symmetry).
- PENDING (2026-11-12): a_profiled_bobyqa_fit_polishes_its_tolerance_stop: main Linux ends 6.6e-7 above L-BFGS on the 4-leg cap (first stop 0.25 above); branch 1.1e-5 (first stop 0.32) vs band 1e-5. Revisit after stage 3 (chaotic BOBYQA path vs EBE rounding).

## Stage 3 progress (2026-11-12, session 3)
- DONE: with_plan/with_bucket_plan (record on miss, no variant cap; eviction past 8), term plans no cap, F64 guard check.
- DONE: run() plan-only (PredLanes, pop model plan, endpoint-only); ChannelPlan/theta_channels (parser detection too), DxSeeds, nn-leaf path in run, kjit_record_taped deleted.
- DONE: run_agq/agq_planned plan-only (node plans per node, endpoint-only, nn); agq_taped, ExactAnchor/Bucket, exact-anchor padding deleted. PK-space lift deleted (+tests). psi_fused/theta_channels tests deleted.
- DONE: canonical_jets/canonical_jets1 plan-only (Jet1/Jet2 plain structs); generic fallbacks deleted.
- DONE coverage: sin/cos PkNum primitives, JetFn Pnorm/Tanh/Logistic (+softplus jet moved to special), ReLU as select, clamp_dual_floor/closed-form init via branch API, interpreter Cmp via br_cmp, nn_val only for non-recording types, Flat registers u32 (register file full).
- DONE: SS Found (Fixed/Iterated/Constant), held coords, sub_depth masks; solver stats tee for plan loops + declined-segment replay; times_have_non_finite generic.
- NEXT: fixed_ebe_score(_iov), fixed_eta_theta_sweeps, fixed_eta_data_score, exact_inner_gradient(_in)/exact_obs_grads_into, ng_jets3/ng_theta_adjoint generic (Taylor/DualH), hessian.rs covariance (DualT/DualH), focei_kernel DualT; then delete vtape/vfree/rtape-nested/dual1/dual2/dual_mixed/taylor; port tests.
