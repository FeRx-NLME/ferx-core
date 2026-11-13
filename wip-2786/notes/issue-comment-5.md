## Prototype step 5: subject-level compiled walks win, bit-identically

This builds the design from the previous comment at subject granularity on **wiov's real data**: 200 subjects, 2 occasions, 2 doses and 22 observations each. The model is wiov.ferx: CL = TVCL·exp(η_CL + κ_occ), V = TVV·exp(η_V), KA = TVKA·exp(η_KA).

There are 6 directions, matching ferx's bucket: 3 η, 2 κ and one padding lane. They are seeded one-hot, as the inner loop's jets are. The propagator is ferx's `propagate_one_cpt_oral_g` body, rate terms included. Its two value branches (`ka ≈ ke`, positive parameters) go through an explicit guard API that the recorder turns into checks; a failed check falls back.

**The walk is recorded once per event structure.** η, θ, time steps and doses are inputs, so subjects with the same sequence of record kinds share one compiled function. The recording:
- applies hash-consing, so an identical operation on identical operands is one node (per-occasion parameters and repeated factors);
- folds the seeds' exact `1.0` and `0.0`;
- skips structurally zero lanes.

It is then lowered to Cranelift.

**Baseline:** the same walk over an LLVM-inlined fixed-width dual (`Fx<6, M>`), as fast as I could make it (see "Fairness" below).

**Realistic setting:** sampling times jittered so every subject's times are unique, as in real data. Three interleaved runs, 3000 repetitions over all subjects; instructions and cycles are for the whole process, compile included.

| | fixed duals (LLVM) | compiled walk (Cranelift) | |
|---|---|---|---|
| first order (6, 0) | 1.21–1.24 µs/subject, 9.6 G instr | **0.66–0.74 µs**, 4.4 G instr | **1.75× faster** |
| second order (6, 21) | 7.2–9.0 µs/subject, 56.5 G instr | **2.58–2.77 µs**, 12.2 G instr | **~2.9× faster** |

- **Compile:** one function for all 200 subjects, 14 ms at first order and 63 ms at second order.
- **Parity:** predictions and every jet lane are **bit-identical** to the fixed walk: 0 of 30 800 lanes differ at first order, 0 of 123 200 at second order.

### What makes it work

- **Specialising once per structure, not per call.** The kernel-level JIT lost because it re-discovered sparsity at every call; here it is discovered once.
- **Sharing by structure, not by values.** Baking times in as constants means one function per subject when times differ, and that was slow: 200 compiles taking 2.1 s (10.7 s at second order), and the walk itself got slower (1.58 µs; 41.7 µs at second order), most likely from 200 functions missing the instruction cache.
- **Sparsity is the source of the gain.** Hand-hoisting the per-occasion parameters into the fixed walk gave nothing (1.44 against 1.45 µs).

### Fairness checks

- **The fixed baseline is sensitive to LLVM inlining.** Adding a second walk variant to the same function made the second-order baseline 14 µs; isolating it restored 7.2–9.0 µs. The table uses the restored, faster baseline.
- **This baseline is the idealised fixed dual.** ferx's own walk carries more machinery, so the gain against ferx may be larger or smaller. That has to be measured inside ferx.

### Not covered yet

- **The outer gradient.** The θ-taped `DualV<6,21>` needs the reverse code generated from the same graph; `ir::reverse` shows how.
- **Datasets with many distinct event structures.** Each needs its own compile (14–63 ms) and code space. A remedy is per-segment functions chained by the walk.
- **ODE models, SS, lag and infusions.**
- **Whole-fit timing.**

Bench: `/scratch/mu/ad/lanebench` (`src/subject.rs`, `src/jit.rs`, `final.sh`).
