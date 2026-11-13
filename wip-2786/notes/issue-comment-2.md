## Prototype step 2: trace and compile kernels with Cranelift at run time

This tests the one design that could address both speed and binary size: compile the engine kernels when the fit starts instead of instantiating them per type × width.

**Method:**
- Run the kernel once on a recording type (`Tr`), which records the operations as a graph.
- Lower the graph to Cranelift at the model's exact width, in one of two modes:
  - **strict:** every lane is computed, with structural zeros as `0.0` constants. Cranelift never folds `x·0`, so the results are bit-identical to the fixed dual.
  - **folded:** lanes known to be zero are not computed. These are the tangents of constants and of value-only inputs, the lanes outside each input's sparsity mask, and everything they propagate to.
- One traced function is one interval: dose, `propagate_one_cpt_oral_g`, readout.
- The baseline is the LLVM fixed-width dual, inlined, with `dt` and the dose made opaque (`black_box`). A compile-time `dt` let LLVM hoist the exponentials out of the walk; ferx's `dt` comes from data, so that was unfair to the JIT.
- **Dense seeds:** every parameter depends on every direction. **Sparse seeds:** each of CL, V and KA depends on its own direction, as when each individual parameter carries one η.

Results are ns per walk, ratio against fixed in parentheses, with instructions in the last column. All 30 runs have bit-identical checksums.

| n, m | fixed | dense, strict JIT | sparse, folded JIT | instructions, sparse (fixed → JIT) |
|---|---|---|---|---|
| 3, 0 | 730 | 1 124 (1.5×) | 919 (1.26×) | 1.12 G → 1.40 G |
| 4, 0 | 854 | 1 322 (1.4×) | 991 (1.16×) | 1.19 G → 1.46 G |
| 6, 0 | 1 279 | 1 747 (1.4×) | 1 048 (**0.82×**) | 1.91 G → 1.56 G |
| 3, 6 | 2 313 | 3 576 (1.6×) | 1 934 (**0.84×**) | 3.37 G → 2.83 G |
| 6, 21 | 6 018 | 17 141 (3.1×) | 2 790 (**0.46×**) | 8.33 G → 4.08 G |

**Trace + compile time:** 1.7 ms at (4, 0); 15–19 ms at (6, 21).

### Reading

- **On identical work, Cranelift's code is 1.4–3× slower than LLVM's.** The gap is not vectorization: LLVM's code is mostly scalar too (1 569 `mulsd` against 49 `mulpd` at (6, 21)). It is optimization and register allocation; Cranelift's straight-line code at (6, 21) runs 3.1× slower in wall time for 1.7× the instructions.
- **Structural sparsity is the JIT's only real lever, and it is large for wide second-order payloads.** With one η per parameter, the folded JIT skips most of the Hessian-payload work and runs 2.2× faster at (6, 21). For small first-order duals it remains 16–26% slower.

### Status

Not yet decided. The kernel-level win is real but narrow (wide second-order payloads), and it comes with a substantial infrastructure:
- recording every engine kernel;
- guards for value-dependent branches (`ka ≈ ke`, zero rates), with re-tracing;
- a cache keyed by kernel, sparsity pattern and branch path;
- integration with the θ tape.

The ODE steppers' adaptive control cannot be traced as straight-line code at all.

Two open questions decide it:
1. **How sparse are real ferx payloads in a fit?** This needs instrumented counts on wiov and pk2.
2. **What share of a fit is wide second-order work?** On wiov, all dual-typed code is 24%, so a 2× win on part of it is a single-digit-percent fit gain.

Bench: `/scratch/mu/ad/lanebench` (`src/jit.rs`, `jitrun.sh`).
