## Prototype step 4: what it costs to specialise per call

In ferx the propagator sees only dual values. A JIT specialised on sparsity must therefore, at **every call**, find which lanes of its inputs are zero and look up the compiled function for that pattern. I added exactly that to the benchmark: scan 5 dual inputs and 2 scalars into a 7-word key, then search the cache. Sparse seeds; per interval step:

| n, m | fixed (LLVM) | folded JIT, pattern known in advance | folded JIT + per-call key |
|---|---|---|---|
| 4, 0 | 36.1 ns | 43.8 ns | **59.2 ns** (1.64×) |
| 6, 0 | 51.4 ns | 44.3 ns | **67.9 ns** (1.32×) |
| 4, 10 | 158.0 ns | 99.7 ns | **128.9 ns** (0.82×) |
| 6, 21 | 240.9 ns | 116.6 ns | **177.8 ns** (0.74×) |

- **Keying costs 15–60 ns per call.** That is the size of a whole first-order kernel call. Even an ideal key (~8 ns) only breaks even at first order.
- **Second order still wins at kernel granularity (0.74–0.82×).** But ferx's second-order types on the hot path (`DualV<4,10>`, `DualV<6,21>`) belong to the outer gradient, whose values are on the θ tape. A JIT'd kernel there must record itself as one tape block with a compiled reverse rule. The `Op::Rhs` / `ir::reverse` machinery does this for the ODE RHS, but it is a substantial integration.

### Conclusion on granularity

At kernel granularity the JIT cannot win where most derivative time is: the first-order inner-loop duals, 10–17% of fit time. Discovering the pattern costs as much as the work it saves.

The JIT would pay only at **subject granularity**, where the pattern is discovered once per trace and the walk is reused across the thousands of inner iterations. The walk is fixed by the subject's schedule and shared across subjects with the same schedule. That requires recording whole event walks, and those branch on values throughout:
- the `ka ≈ ke` fallback;
- zero clearance;
- steady-state equilibration loops with a value-dependent iteration count;
- lag times that reorder events.

A sound trace needs those branches made explicit to the recorder: a branch/select API in the engine kernels, as JAX requires `lax.cond`. That is an engine-wide refactor.

An in-ferx kernel-level build would mostly reproduce the first-order slowdown above on wiov, so I stopped before building it.

Bench: `/scratch/mu/ad/lanebench` (`KEYED=1`, `keyed.sh`).
