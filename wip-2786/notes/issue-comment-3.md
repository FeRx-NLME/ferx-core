## Prototype step 3: how sparse ferx's duals really are, and two ways to exploit it

### Measured sparsity in real fits

Throwaway instrumentation, not committed, on `DualV`'s multiply, divide and unary bodies; `DualF` runs the same bodies. Release build without LTO, the benchmark fits run to convergence.

"Multiply terms live" is the share of a product's dense terms (`2` per tangent, `4` per payload entry) whose factors are not exactly zero, i.e. the work a fold that skips zeros would still do.

| Fit | Type | Ops | Zero tangent lanes | Zero payload lanes | Multiply terms live |
|---|---|---|---|---|---|
| pk2 | `<4,0>` | 24.5 M | 29% | – | 53% |
| pk2 | `<4,10>` | 2.4 M | 30% | 37% | 39% |
| wiov | `<6,0>` | 107.7 M | 76% | – | **27%** |
| wiov | `<6,21>` | 13.2 M | 74% | 84% | **14%** |
| pk2_ode | `<4,0>` | 149.0 M | 60% | – | 41% |
| pk2_ode | `<4,10>` | 17.9 M | 48% | 49% | 37% |
| mm | `<3,0>` | 38.0 M | 39% | – | 31% |
| mm | `<4,10>` | 2.3 M | 39% | 47% | 16% |

Between half and six-sevenths of the derivative arithmetic multiplies exact zeros. In wiov this is structural:
- the 6 directions are 3 η plus κ_CL for each of 3 occasions;
- one occasion's κ is dead in the others;
- CL depends on 2 directions, V and KA on 1 each.

### Run-time masks in LLVM code: rejected

I tried a fixed-width dual carrying a live-direction bitmask: operations compute only the live lanes, with a dense fast path when every direction is live. It is **3–10× slower** than the plain fixed dual, even on dense seeds.
- Constants and fresh state start with no live directions, so most operations mix masks and the dense fast path rarely runs.
- Bit-iteration loops branch on data, do not vectorize, and push values out of registers.

At n ≤ 8 a run-time zero check costs more than the arithmetic it skips. Sparsity can only pay when it is resolved **before** the code runs.

### The JIT with trace-time sparsity: an estimate, not a measurement

Combining the live-term fractions above with the measured Cranelift penalty on identical work (1.4–1.6× first order, ~2.3× at (6,21)) and each type's share of fit time from the profiles:

| Fit | Second-order share | First-order share | Estimated fit saving |
|---|---|---|---|
| pk2 | 8.6% | 11.5% | ~6% |
| wiov | 3.2% | 15.3% | ~12% |
| pk2_ode | 11.2% | 16.9% | ~12% |
| mm | 7.4% | 10.0% | ~11% |

**These are upper bounds.** They assume:
- every exact zero is known at trace time;
- all dual-typed time sits in traceable kernels;
- tracing, guards and caching cost nothing.

ODE stepper control and the tape sweeps are not straight-line code.

### Where this leaves the design

- **Rejected:** runtime-width lanes (step 1) and run-time masks (this step). At ferx's widths, inlined fixed-width LLVM duals beat every run-time representation.
- **Open, with an upper bound of roughly 6–12% per fit plus a large binary reduction:** trace each kernel and compile it with Cranelift, specialised per sparsity pattern. The cost is a large new infrastructure:
  - a recording type through every engine kernel;
  - guards and re-tracing for value-dependent branches;
  - a cache keyed by kernel, sparsity pattern and branch path;
  - θ-tape integration;
  - keeping f64 predictions bit-identical (the strict mode shows this is possible).

That is a design decision for the maintainers, not something to start autonomously.

Data: `/scratch/mu/ad/stats/*.txt`, bench `/scratch/mu/ad/lanebench` (`src/masked.rs`, `src/jit.rs`).
