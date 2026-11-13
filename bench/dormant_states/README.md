# Dormant states in the exact linear path (#2760)

**Question.** What does a linear ODE model cost when most of its states are idle in any
one segment, and what does leaving them out of the matrix exponential buy?

**Fixture.** `sim_urine_intervals.py` writes a dataset and a ferx model for any number of
collection intervals `K`:

- **Dosing:** a parent drug, infused over 1 h at the start of each of `K` 24-h intervals.
- **Elimination:** cleared renally and by formation of one metabolite, which is excreted in
  urine.
- **Observations:** plasma parent at 1, 2, 6 and 12 h after each dose, plus the amounts of
  parent and of metabolite excreted over each interval, observed at its end. An `EVID=2` row
  then advances the interval index `IDX`.
- **Size:** 24 subjects. The data are deterministic for a given seed.

ferx has no per-compartment reset, so the model collects each interval in its own
accumulator, `d/dt(u_k) = G_k · k · x` with `G_k = if (IDX == k) 1 else 0`. That gives
`2 + 2K` states, of which at most four move in any segment. `urine_intervals_k8.ferx` and
`urine_intervals_k8.csv` are the committed `K = 8` (18-state) case.

**The truth is independent of that formulation.** The generator simulates the reduced
4-state system (parent, metabolite, and the two cumulative urine amounts) exactly: a matrix
exponential by scaling and squaring, which `--check` compares against a fine-step RK4
(3.8e-14). An interval's urine is the difference of the cumulative amounts at its ends.
At the true typical values, ferx's PRED on the 18-state model matches it on all 1152
records to the sdtab's printed precision (worst 4.97e-7 absolute, the 6-decimal rounding).

```bash
python3 bench/dormant_states/sim_urine_intervals.py --k 8            # this directory
python3 bench/dormant_states/sim_urine_intervals.py --k 14 --out /tmp
python3 bench/dormant_states/sim_urine_intervals.py --k 3 --check
cargo run --release -p ferx-cli -- bench/dormant_states/urine_intervals_k8.ferx \
  --data bench/dormant_states/urine_intervals_k8.csv --threads 1
```

## Results

Linux x86_64 (Xeon Platinum 8488C), `--threads 1`, release build, `main` at c5dac24b and the #2760 branch interleaved, on a shared box (load 4–6 from other work during the run). The fixture is `bench/dormant_states/` (24 subjects; `K` collection intervals, `2 + 2K` states).

One evaluation (`maxiter = 0`), median of 3:

| `K` | states | `main` c5dac24b | with #2760 | speed-up | outputs |
|---|---|---|---|---|---|
| 2 | 6 | 0.70 s | 0.41 s | 1.7× | byte-identical |
| 4 | 10 | 3.38 s | 1.15 s | 2.9× | byte-identical |
| 8 | 18 | 22.33 s | 4.68 s | 4.8× | byte-identical |
| 14 | 30 | 136.55 s | 29.03 s | 4.7× | byte-identical |

Full fits, covariance step on, one run each:

| `K` (states) | | total | estimation | covariance | iterations | OFV |
|---|---|---|---|---|---|---|
| 4 (10) | `main` | 25.2 s | 23.2 s | 1.86 s | 45 | 436.9229 |
| 4 (10) | with #2760 | 7.8 s | 7.1 s | 0.70 s | 45 | 436.9229 |
| | speed-up | 3.2× | 3.3× | 2.6× | | sdtab + fit YAML byte-identical |
| 8 (18) | `main` | 180.6 s | 169.9 s | 10.20 s | 50 | 678.4511 |
| 8 (18) | with #2760 | 30.1 s | 27.7 s | 2.28 s | 50 | 678.4511 |
| | speed-up | 6.0× | 6.1× | 4.5× | | sdtab + fit YAML byte-identical |

Every bundled ODE example the release CLI runs, `main` at e6244a7b (with #2747) against the #2760 branch at 1f87321a, full fits, median of 3, the same box idle. Few states, so little to win: the point is what moves and what does not.

| example (bundled data) | exact linear path | `main` e6244a7b | with #2760 | outputs |
|---|---|---|---|---|
| `one_cpt_iv_ode` | yes | 0.064 s | 0.063 s | byte-identical |
| `two_cpt_iv_ode` | yes | 0.172 s | 0.178 s | byte-identical |
| `three_cpt_iv_ode` | yes | 0.920 s | 0.917 s | byte-identical |
| `warfarin_ode` | yes | 0.140 s | 0.141 s | byte-identical |
| `transit_2cpt` | yes | 1.006 s | 0.984 s | byte-identical |
| `warfarin_ode_bloq` | yes | 0.108 s | 0.108 s | byte-identical |
| `warfarin_ode_lagtime` | yes | 0.424 s | 0.427 s | byte-identical |
| `pkpd_nonmem` | yes | 0.132 s | 0.122 s | byte-identical |
| `bioavailability_ode` | yes | 0.188 s | 0.176 s | sdtab byte-identical; YAML differs only in `final_gradient_max_abs` (1.76759e-6 vs 1.76808e-6) and a correlation's 16th digit; the same mechanism as `emax_pkpd`, see the note below the table |
| `emax_pkpd` | yes | 0.103 s | 0.095 s | a different path to the same optimum: 127 → 135 iterations, OFV equal, estimates ≤ 2e-6 relative. At the start (`maxiter = 0`) sdtab and OFV are bit-identical; the outer gradient is not, at 7.9e-13 relative on Linux (6.5e-12 on macOS). Not rounding: see the note below the table |
| `mm_iv` | no | 0.139 s | 0.139 s | byte-identical |
| `mm_oral` | no | 0.648 s | 0.653 s | byte-identical |
| `tests/fixtures/mm_basin_shift` (200 subjects) | no | 14.33 s | 14.40 s | byte-identical |
| `warfarin_sde` | no | 35.46 s | 35.86 s | byte-identical |

The `[markov_model]` and `[event_model]` examples need feature builds the CLI does not have, and were not run.

### Why `emax_pkpd` takes a different path

Measured on Linux x86_64 (the reference platform), then on macOS arm64, in a
debug test build. The fits were run through `run_model_with_data` with
`optimizer_trace = true`, the trace printed at full precision, and process-wide
copies of the test switches plus counters on the spectral path. None of that is
committed. Two Linux runs gave bit-identical traces.

**Where the fits part.** At the first gradient, at the start point itself. The
parameter values first differ at the next trace row. The worst gap there is
7.9e-13 as `|a − b| / (1 + |b|)` on Linux (on `TVV`; 6.5e-12 on macOS, on
`TVCL`), and all 12 coordinates differ. On the two gated fixtures the same
measure is 7.0e-17 and 4.2e-16. So this is not the fold-order rounding those
fixtures bound.

**Why.** The spectral propagator (`spectral_of`) builds `A`'s coupling graph
from structural zeros. `A` is lower-triangular here (depot → central →
effect), so with its zeros structural every compartment is acyclic:

- its eigenvalue is snapped to the exact diagonal;
- its projectors are masked to their exact support.

On `main` the recorded RHS block kept `A`'s zeros as tape nodes. On the taped
gradient, `A` therefore read as one cyclic 3×3 block, with eigenvalues from the
eigensolver and no masks. The `f64` objective sees the zeros as structural and
takes the acyclic form. So on `main` the gradient and the OFV ran two different
constructions of the same propagation. With #2760 the tape drops those zeros,
and both run the acyclic form.

**Evidence.** All gradients at the start point. "Kept" is `RHS_DEAD_KEPT`;
"block off" is `RHS_BLOCK_OFF`, the per-operation tape; "Padé" is
`SPECTRAL_OFF`.

| comparison | Linux: worst gap (coordinates) | macOS: worst gap (coordinates) |
|---|---|---|
| `main` (c61aa80a) vs this branch, kept | 0 (0 of 12) | 0 (0 of 12) |
| this branch vs kept | 7.9e-13 (12 of 12) | 6.5e-12 (12 of 12) |
| this branch vs kept, both on Padé | 0 (0 of 12) | 0 (0 of 12) |
| this branch vs block off | 3.6e-13 (4 of 12) | 2.4e-16 (1 of 12) |
| this branch vs block off, both on Padé | 1.9e-16 (1 of 12) | 1.9e-16 (1 of 12) |
| spectral vs Padé: this branch / kept | 2.8e-12 / 2.3e-12 | 2.8e-12 / 3.7e-12 |

The counters on `spectral()` count the taped calls whose coupling graph has a
cycle. On Linux: kept 387 of 390, this branch 0 of 414, block off 0 of 402.

What the rows show:

- Kept is `main`'s recording bit for bit: the same trace in every column but
  wall time, 127 iterations on both.
- The counters show the mechanism directly: with the zeros kept, nearly every
  taped call reads `A` as cyclic; with them dropped, none does.
- With Padé everywhere, dropping the zeros changes no bit at the start point,
  so the whole start gap comes from the spectral path's graph.
- This branch and block off run the same acyclic construction. Their 1.9e-16
  operation-order difference (the Padé row) comes out of the spectral path as
  3.6e-13 on Linux. So on Linux the gap's size alone does not separate the
  graph change from rounding; the counters and the Padé row do.
- The gap is the same order as the spread between the engine's two exact
  propagators.
- No state goes dormant on this model, so the dormant reduction plays no part.

For scale: on Padé alone, where only operation order differs, the two fits
first part at the 7th trace row (2.3e-16) and finish one iteration apart
(129 and 128) at the same OFV.

The Linux iteration counts, 127 (`main`) and 135 (this branch), are the release
CLI's above. The macOS debug build gives 126 and 131. Both are the path change.

**`bioavailability_ode`** moves by the same mechanism, at a smaller size. Its
start gradient differs by 1.2e-14 on Linux (7 of 7 coordinates; 1.1e-14 on
macOS) and by 0 on Padé. Kept reads `A` as cyclic in 1380 of 1400 taped calls,
this branch in none. The fits take 67 iterations in every variant, and the
final traced max |gradient| moves from 5.921093e-6 (`main` and kept) to
5.918084e-6. That is the moved `final_gradient_max_abs` in the table above.
