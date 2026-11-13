# #2786 — compiled per-subject recordings replace the dual-number AD (WIP backup)

Branch `perf/runtime-width-lanes`. This folder is a work-in-progress backup (plan, notes,
helper scripts) and is **removed before the PR**. Snapshot taken 2026-11-13 before a server
upgrade; the code is mid-change (see "State at snapshot").

## Goal (user's requirements)

- One PR replacing ferx-core's dual-number AD (DualV/vtape, DualF, DualT/DualH, Dual1/Dual2/
  DualMixed, Taylor-as-AD) with compiled per-subject recordings ("plans",
  `src/parser/kernel_jit*.rs`). Every derivative consumer goes through plans; dual types and
  generic fallbacks deleted.
- Optimal implementation, no matter how big the rewrite; re-derive from the mathematics when
  something fails. Exact analytic derivatives only — never finite differences as a fix or as
  the validation oracle (validate against an independent dual: `sens::vfree::DualF`, now a
  test-only forward jet).
- 100% coverage; faster than main on the large sets (small fits may be slightly slower);
  ferx-r / public-API breakage is acceptable.
- Commit only when asked. Incremental unoptimized (`dev`) builds; optimized only to benchmark.

## Architecture (as built)

- `Tr` (non-generic now) records into a `Recording`; `Recording::plan` cuts it into chunks
  (shared by structure, `ChunkCache`), loop calls (`CallKind::{Ode, Fixed, Repeat}`), tiers
  (interpret_func → native Cranelift tiers 1–3, background compile; all tiers bit-identical).
- Order gates (`Flat::Gate`) give iterative refinements exactly the steps a layout's
  derivative order needs (LU refinement `REFINE_STEPS`, spectral chord steps).
- Value reads are forbidden in a recording (`Tr::val` fails it): every decision goes through
  the branch API (`br_cmp`, `br_cmp_c`, `br_abs_lt`, `br_is_nan`, `select_if`), jet
  primitives (`JetFn`: NegLogNormCdf, Softplus, Pnorm, Tanh, Logistic, LogNormCdf, InvMills,
  ExpM1) or host functions (`HostFn`).
- Plans: predictions (`with_plan`, `plan_forward`, runtime width now — no bucket padding),
  subject terms (`sens/ode_adjoint_terms.rs`: TTE, CTMM, DTMM, binary, whole non-Gaussian),
  and NEW kernel plans (`sens/ode_adjoint_kernel.rs`): the likelihood kernel recorded with
  every input as a parameter slot (`Leaf`/`Inputs`), replayed per call, one multi-seed
  reverse for every output's adjoint. `SubjectKey::with_variant` keys kernel configurations.

## Done (this and earlier sessions)

- Predictions, inner gradient, outer gradient prediction side, AGQ nodes, NG term plans.
- NG curvature in the FOCE/FOCEI marginal: one exact plan Hessian of the whole term
  (`survival::non_gaussian_eta_jets2`); FD stencils, `EtaHessianRoute`,
  `guarded_markov_fd_hessian`, `data_term_hessian_steps` deleted (`data_term_hessian_fd`
  kept `#[cfg(test)]` as the Richardson reference for closed-form cells only).
- vtape (DualV) deleted; DualF test-only (`#[cfg(test)] mod vfree`); eta_jets deleted;
  NN kept-forward/VJP API deleted; rhs_jit VJP and RhsKernel tape hook deleted.
- `Tr` non-generic; `with_plan`/`kjit_record`/`plan_forward` runtime width.
- Pre-start TTE times: `ode::predictions::fill_prestart_states` generic, used by the dual
  walk (`sens::ode_provider::integrate_g`) too.
- Serial residual correlation: `SerialShape` (structural pairing + `C(ρ)` in T arithmetic).
- Kernel plans (UNVERIFIED, see below): `run` (outer gradient), `fixed_ebe_score{,_iov}`,
  `gaussian_fixed_eta_data_score`, `fixed_theta_data_eta_grad`, `agq_planned` (node
  predictions as leaves, node locations as outputs, two-phase reverse).
- Binary size (ci-fast, measured): main 72.7 MB code → branch 50.1 MB (before today's
  deletions); expected final ≈ 36–38 MB.

## State at snapshot — the open bug being chased

Kernel plans return the **same output whatever their inputs** on replay (diagnosed with the
TEMP `FERX_TMP_KVERIFY` check in `ode_adjoint_kernel.rs::kernel_run`: fresh recording 3534.9 /
4503.3 / 4523.6 vs plan 3227.1 every time) → wrong gradients in fixed_eta_gradient,
gauss_newton, mixture, outer_optimizer tests (~30 failures in those groups).
Ruled out so far: `LoadParam` masks (FREE), chunk literals, `forward_layout`/`forward_kept`
input copy, `ctx.params` reads (structural only), `seen_value`. Next to check:
1. The dumped kernel chunk (`FERX_KJIT_DUMP=1 tools/onetest.sh the_fixed_eta_data_score_is_exact`)
   — whether the chunk's inputs are the parameter slots or recorded literals (are the leaves
   reaching the code as `LoadParam`, or did `Tr::theta` leaves get hash-consed / folded?).
2. The plan cache key: kernel `SubjectKey` = (model, subject, n=0, n_in, kind 8, variant) —
   check `lookup`'s thread-local LAST cache and whether a different kernel shares the key.
3. `Plan::forward_logged` with `n_eta = 0` and `n_theta = n_in` (large slot counts).

## Remaining work (in order)

1. Fix the kernel-plan replay bug above; re-run `tools/declines.sh fixed_eta_gradient
   gauss_newton mixture outer_optimizer`; then full suite (`tools/libtest.sh`).
2. Covariance Hessian on plans (`sens/ode_adjoint_hessian.rs`, ~1800 lines on
   `DualT<K,R>`/`DualH`): Hessian-vector products = prediction plan forward at tangent
   layout (K directions in (θ, η)) → kernel plan forward with tangent lanes (packing chain
   on x recorded inside) → kernel reverse → prediction plan reverse; θ chain curvature
   `Σ ∂Ψ/∂θ_m·θ_m″`. Covers AGQ exact anchor and the NN reduction; drops HESSIAN buckets.
3. `run`/`run_agq`/fixed-EBE still dispatch over const-N `WIDTH_BUCKETS` with padding →
   runtime width; `Recording::plan`'s 63-η limit → multi-word direction masks.
4. Retire rtape (`R`, `DualT`, `DualH`) — keep a small forward nested-dual helper only if
   the kernel's `d2_seed` still needs it — and `Dual1`/`Dual2`/`DualMixed` (inner optimizer,
   bayes, saem, hmc, provider) by moving those consumers onto plans.
5. Open defects: rodas5p 4e-7 fixed-η θ-adjoint vs forward dual
   (`the_fixed_eta_theta_adjoint_is_the_forward_theta_tangents`); BOBYQA profile test
   (`a_profiled_bobyqa_fit_polishes_its_tolerance_stop`, Δ 1.1e-5 vs main 6.6e-7 — measure
   objective noise along a line near the optimum, main vs branch, before touching it);
   fixed-η score's FD fallback (`obs_nll_subject_grad_impl`) → exact-only.
6. Cleanup: TEMP code (`FERX_TMP_TIER`, `FERX_TMP_TIERCHECK`, `FERX_TMP_KVERIFY` +
   `Tr::v_dbg`, `TIERBITS`, `FERX_TMP_NORESET`, GLOG/guard_log_diff, record_sub debug
   print, `LaneProg::n_out/code`), `private_interfaces`, dead-code sweep
   (`tools/deadsweep.py`), delete `wip-2786/`.
7. Release: full lib suite, `tests/*.rs`, nn/markov feature sets, `tools/preflight.sh`
   (clippy -Dunused incl. --lib), coverage, docs, changelog fragment,
   `tools/update-public-api.sh`, benchmarks vs main (pk2, pk2_ode, wiov, mm; interleaved
   A/B), then the PR (fill `.github/PULL_REQUEST_TEMPLATE.md`).

## Tools (`wip-2786/tools`, originally `/scratch/mu/ad/kj`)

Paths inside are absolute to the old box (`/root/package/ferx-core/.worktrees/lanes`,
`/scratch/tgt-lanes-test`, `/scratch/mu/ad/...`); adjust after the move.
- `libtest.sh` — build + full lib suite (`--features survival,markov,nn`, dev profile,
  `CARGO_PROFILE_DEV_DEBUG=0`, target `/scratch/tgt-lanes-test`).
- `onetest.sh <filter> [secs]` — one test with output (`FERX_KJIT_DEBUG=1` for recorder
  decline reasons).
- `declines.sh <groups…>` — groups with recorder debug; distinct decline reasons + results.
- `rmitem.py FILE SIG…` — delete Rust items by signature; `deadsweep.py`; `sizes.py BIN…` —
  binary size by number type.
- Notes: `notes/audit_formulations.md` (gate/formulation audit), `notes/issue*.md`.
