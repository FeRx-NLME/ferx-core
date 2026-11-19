# Modeled-infusion ODE model: compiled plans vs Dual2 (#3010)

**Question.** #3010 reported that on an ODE model whose infusion end moves with η, the
compiled per-subject plans (#2786) cost more than the Dual2 path they replaced: 479 s and
3.2 GB against 160 s and 96 MB. #3011, #3031, #3032 and #3033 then went after that cost. The
reproducer data was lost, so the combined effect had not been measured. Is there still a gap?

**Fixture.** This fixture follows the issue's description. `sim_modeled_infusion.py` writes
the data and `modeled_infusion.ferx` is the model:

- **Model:** a 2-compartment linear `[odes]` model (`depot`, `central`, `peri`) with
  first-order absorption. 8 THETAs (two of them `WT` exponents), 4 ETAs (`CL`, `V1`, `D1`,
  `F1`), proportional error.
- **Dosing:** 28 doses q24h into the depot as a **modeled-duration** zero-order input
  (`RATE=-2`), with `D1 = TVD1·exp(η)` and `F1 = exp(η)`.
- **Size:** 156 subjects, 7176 records, 2808 observations. `WT` is constant within a subject.
- **Designs:** `--design random` (the default) puts 70 % of the 18 samples 0.25–6 h after a
  dose, so infusion ends near 2 h fall on both sides of many samples. `--design fixed` uses
  fixed times after doses 1, 2, 14, 27 and 28.
- **Starting point:** the model starts away from the truth (CL ×2, V1 ×½, KA ×0.4,
  D1 ×2, ω²_D1 = 0.4), so the EBE searches move the ends across the samples.

The data come from RK4 integration (dt ≤ 0.01 h, cut at every infusion end, dose and
sample) outside ferx. They are standard library only and seeded.

**The fixture is the right shape.** The largest single recording in a fit on the random
design is 113k instructions. The issue reported about 122k for its subject with infusions.

```bash
python3 bench/modeled_infusion_3010/sim_modeled_infusion.py > /tmp/mi_random.csv
python3 bench/modeled_infusion_3010/sim_modeled_infusion.py --design fixed > /tmp/mi_fixed.csv
cargo build --profile ci-test -p ferx-cli
/usr/bin/time -l target/ci-test/ferx bench/modeled_infusion_3010/modeled_infusion.ferx \
  --data /tmp/mi_random.csv --threads 1
FERX_KJIT_DEBUG=1 target/ci-test/ferx ...   # adds the plan layer's counters at the end
```

## Results

Measured on macOS arm64 (Darwin 25.5) with the `ci-test` profile, FOCEI and a full fit
including the covariance step, one run each. Builds:

- `main` is at 96da60f5, after #3011, #3031, #3032 and #3033.
- Dual2 is at 4ff6a9d4, the issue's baseline from before #2939.

Wall time is from `/usr/bin/time -l`, and peak RSS is its "maximum resident set size". Every
run converged, and both builds reach the same OFV.

| design | threads | `main` wall | `main` peak RSS | Dual2 wall | Dual2 peak RSS | OFV |
|---|---|---|---|---|---|---|
| random | 1 | **7.9 s** | **270 MB** | 63.9 s | 338 MB | −6962.5748 |
| random | 8 | **2.2 s** | **500 MB** | 10.3 s | 2021 MB | −6962.5748 |
| fixed | 1 | **10.6 s** | **205 MB** | 85.0 s | 340 MB | −7453.6859 |
| fixed | 8 | **2.5 s** | **426 MB** | 13.4 s | 2320 MB | −7453.6859 |

**Conclusion: no gap is left on this shape.** `main` is 5–8× faster than Dual2. Its peak
memory is lower too: by 20–40 % on one thread, and by 4–5× on eight. Dual2's per-thread dual
sweeps grow with the thread count, while the plans and compiled kernels are shared between
threads.

The issue's own Dual2 figure was 96 MB. Dual2 needs about 340 MB here on comparable data, so
the issue's figures should not be compared with these. Only the same-data pairs above are a
comparison.

### What `main`'s peak holds

These numbers come from temporary instrumentation (not committed) at the end of a
single-thread fit on the random design. About 80 MB of RSS is already present before the
first evaluation (data, model, runtime), and the plan layer adds the rest:

| holder | size |
|---|---|
| chunk kernels' IR (`Tiered::f`), 2.7M instructions × 16 B | ~45 MB |
| of which: interpreted only (tier 0), so it is what runs | 2.6M instructions |
| plans' slot tables (`Call::Chunk::idx`), 5.2M entries × 4 B | ~21 MB |
| chunk programs' code (`ChunkProgram::code`), 0.62M × 24 B | ~15 MB |
| templates (`raw`, `trace`, `rec.code`) | < 3 MB |

Counters for the same fit: 457 recordings, 219 compiled modules and 2393 guard failures.

Two follow-ups were measured and **not** pursued:

- **Sharing slot tables through `Arc<[u32]>`.** A template serves only a small part of the
  resident plans, so sharing its tables with the plans it serves saved only 0.5M of the 5.2M
  entries.
- **Interning identical slot tables.** There are only about 1M distinct entries, so
  interning could save about 15 MB. That is inside the run-to-run RSS noise at this size.

The tier-0 IR cannot be dropped, because the interpreter runs it.
