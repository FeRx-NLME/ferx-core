# Reproduce the biphasic inverse-Gaussian anchor (#388)

Self-contained run guide for ferx's pathway-fraction mechanism
(`FR1*igd(...) + FR2*igd(...)`). Two inverse-Gaussian absorption pathways (fast +
slow) feeding a 1-cpt central compartment, split by a dose fraction. All inputs are
committed in this repo; see `README.md` for the broader anchor catalogue and the
RESULT comparison.

## Files

| File | What it is |
|------|------------|
| `nonmem_anchor/freijer_biphasic_ig.ctl` | **NONMEM control** (ADVAN13 TOL=9, `$DES` sums two IG pathways by `FR1`/`1-FR1`). |
| `nonmem_anchor/biphasic_ig_oral.csv` | **Dataset for NONMEM** — dose CMT 1 (inert, F1=0), obs **CMT 2**. |
| `nonmem_anchor/simulate_biphasic_ig_data.py` | Reproducible generator (pure stdlib, seed 7) — re-run to regenerate the CSV byte-for-byte. |
| `nonmem_anchor/biphasic_ig_fit.ferx` | The matching **ferx** model. |
| `data/biphasic_ig_oral.csv` | Same data re-keyed to obs **CMT 1** (ferx's single-state model). |
| `examples/biphasic_igd_absorption.ferx` | Standalone, self-simulating example (`--simulate`). |
| `nonmem_anchor/results/freijer_biphasic_ig.*` | The committed NONMEM outputs (`#OBJV = −754.211`). |

## Run NONMEM

```bash
cd nonmem_anchor
nmfe75 freijer_biphasic_ig.ctl freijer_biphasic_ig.lst
```

## Run ferx (cross-check)

```bash
# objective at NONMEM's optimum (the anchor), or a free fit:
cargo run --release -- nonmem_anchor/biphasic_ig_fit.ferx --data data/biphasic_ig_oral.csv
# the slow-gated acceptance test:
cargo test --test biphasic_igd_nonmem_anchor --features slow-tests
```

## Regenerate the dataset

```bash
python3 nonmem_anchor/simulate_biphasic_ig_data.py > nonmem_anchor/biphasic_ig_oral.csv
```

## Expected (matched, biphasic-truth data → recovery + agreement)

20 subjects, single 100 mg oral dose, samples 0.25–24 h (240 obs). NONMEM
MINIMIZATION SUCCESSFUL, `#OBJV = −754.211`; ferx's FOCEI objective at NONMEM's
optimum = −754.2113 (~1e-5 agreement). Recovered fixed effects vs truths:

| Param | NONMEM | Truth |
|-------|-------:|------:|
| CL (L/h) | 5.366 | 5.0 |
| V (L) | 56.94 | 50.0 |
| FR1 (fast fraction) | 0.6435 | 0.6 |
| MAT1 (h, fast) | 0.5281 | 0.5 |
| MAT2 (h, slow) | 4.124 | 4.0 |
| CV2_1 | 0.2188 | 0.2 |
| CV2_2 | 0.3453 | 0.5 |
| ω²(CL), ω²(V) | 0.0480, 0.0429 | 0.09, 0.09 |
| σ proportional (SD) | 0.1579 | 0.15 |

The `MAT1 < MAT2` bounds in both the `.ctl` and `biphasic_ig_fit.ferx` break the
pathway-label symmetry — keep that convention when comparing.
