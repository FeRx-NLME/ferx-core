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

The bundled linear-ODE examples (all six on the exact path; few states, so little to win: the point is that nothing is lost), full fits, median of 3:

| example (bundled data) | `main` c5dac24b | with #2760 | outputs |
|---|---|---|---|
| `one_cpt_iv_ode` | 0.08 s | 0.08 s | byte-identical |
| `two_cpt_iv_ode` | 0.37 s | 0.37 s | byte-identical |
| `three_cpt_iv_ode` | 1.42 s | 1.42 s | byte-identical |
| `warfarin_ode` | 0.28 s | 0.28 s | byte-identical |
| `transit_2cpt` | 0.97 s | 0.94 s | byte-identical |
| `bioavailability_ode` | 0.29 s | 0.28 s | byte-identical |
