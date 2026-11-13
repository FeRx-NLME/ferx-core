## Prototype step 1: runtime-width forward lanes fail, by 6–30×

Standalone benchmark of ferx's `propagate_one_cpt_oral_g` (same operations in the same order, a 24-interval, 3-dose walk). It compares today's fixed-width inline dual against a runtime-width value: a `Copy` handle into per-thread scratch lanes, where a constant allocates nothing. Instructions are from `perf stat`, one core, 200k walks. The results are **bit-identical** in every configuration.

| n, m | fixed | lanes (thread-local) | lanes (global, best case) |
|---|---|---|---|
| 2, 0 | 612 M | 18 459 M (30×) | – |
| 4, 0 | 1 011 M | 20 027 M (20×) | 11 700 M (11.6×) |
| 6, 0 | 1 544 M | 23 246 M (15×) | 16 418 M (10.6×) |
| 6, 21 | 11 517 M | 70 507 M (6.1×) | 64 156 M (5.6×) |

Wall time shows the same ratios.

**Why.** An inlined fixed-width dual costs about 5 instructions per operation at n = 4:
- its lanes live in registers;
- LLVM deletes the zero tangents of every constant at compile time.

A runtime-width value round-trips memory on every operation (allocate, load both operands, store) and cannot fold structural zeros. Removing the thread-local lookup altogether (the "global" column) still leaves it 6–11× behind. This is the representation's floor, not an implementation detail.

**Consequence for the design.** Point 1 (runtime-width forward lanes replacing the fixed-width types) is **rejected**. Fixed widths are not the overhead; they are what makes forward mode cheap at ferx's widths (n_η ≤ 8). This agrees with #2481: the modes and formats on the hot path are close to optimal.

The binary-size cost (142 types, ~42 MB) is therefore a size problem, not a speed problem. The remaining speed targets are where the profiles put them:
- the reverse tape sweeps: `vsweep`/`push_cse`, 15–20% of pk2 and pk2_ode;
- out-of-line copying of wide values: `DualV<6,21>` in wiov, ~9%.

Next: examining whether the tape sweeps are at their floor.

Bench: `/scratch/mu/ad/lanebench` (`run.sh`, `quick.sh`).
