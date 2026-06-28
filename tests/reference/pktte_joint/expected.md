# Joint PK-TTE anchor — expected values (#564)

Cross-tool reference for ferx's ODE-accumulated joint PK-TTE fit. All tools fit the
**identical** model + dataset (`pktte_joint.csv`): oral 1-cpt PK (concentration on CMT 2)
+ drug-driven hazard `h = H0·exp(BETA·Cc)`, `Cc = central/V`, accumulated as an ODE state,
with the event on CMT 3. N=120, 78 events / 42 right-censored (~35%).

- ferx:    `pktte_joint_fit.ferx`  (FOCEI)
- nlmixr2: `nlmixr2.R`             (FOCEI; cross-checked with BOBYQA outer optimizer)
- NONMEM:  `nonmem.ctl`            (NONMEM 7.6.0, `METHOD=COND LAPLACE INTER`)

Full NONMEM record (control-stream corrections, covariance, diagnostics): `nonmem_run_summary.md`.

## Estimates

| Parameter | Truth | ferx FOCEI | nlmixr2 FOCEI | nlmixr2 BOBYQA | NONMEM LAPLACE |
|-----------|------:|-----------:|--------------:|---------------:|---------------:|
| CL        | 1.0   | _pending_  | 1.026         | 1.029          | 1.031          |
| V         | 10.0  | _pending_  | 9.946         | 9.951          | 9.897          |
| KA        | 1.0   | _pending_  | 0.9935        | 0.9943         | 0.9796         |
| H0        | 0.015 | _pending_  | 0.02100       | 0.02206        | 0.01491        |
| BETA      | 0.25  | _pending_  | 0.2014        | 0.1930         | 0.2506         |
| ω²(CL)    | 0.09  | _pending_  | 0.0803        | 0.0801         | 0.0814         |
| prop. sd  | 0.10  | _pending_  | 0.1023        | 0.1023         | 0.0919         |
| −2LL      | —     | _pending_  | −1589.60      | −1589.68       | 626.65 (OFV)†  |

(nlmixr2 estimates are on the natural scale: `CL = exp(lcl)`, etc. NONMEM prop. sd = √σ²,
σ² = 0.00844 → 0.0919.)

† NONMEM 7.6.0, `METHOD=COND LAPLACE INTER`, run 2026-06-28: MINIMIZATION SUCCESSFUL,
3.6 significant digits. **The OFV is not cross-tool comparable to nlmixr2's −2LL:** 626.65 is
*without* the `N·log(2π) = 882.18` constant (1508.83 with it), and the tools normalize the
F_FLAG TTE-row likelihood differently — plus LAPLACE ≠ FOCEI on the non-Gaussian rows. For a
likelihood-based cross-check, compare ΔOFV vs Δ(−2LL) between nested models (e.g. BETA fixed to
0); the tool-specific constants cancel in the difference.
**Covariance:** the default sandwich aborts — the R matrix (Hessian) is algorithmically
singular along the flat (H0,BETA) ridge, which `UNCONDITIONAL` cannot fix. `$COVARIANCE
MATRIX=S UNCONDITIONAL` (gradient cross-product) does yield SEs: RSE CL 2.8%, V 1.2%, KA 1.9%,
ω²(CL) 14%, prop.var 6.6% (all well estimated); **H0 27%, BETA 18%, corr(H0,BETA) = −0.91** —
the ridge showing through. Because R is singular, these S-based SEs likely *understate* the true
H0/BETA uncertainty — treat them as a lower bound (a profile likelihood or bootstrap would be
honest). Diagnostics: ETABAR p = 0.60, η-shrinkage 4.0%, ε-shrinkage 11.8%.

## Notes

- **The PK block recovers cleanly and agrees across all three tools.** CL, V, KA, ω²(CL),
  and the proportional error match to ~2–3 significant figures. This is the valid
  cross-tool numerical check, and what ferx must reproduce.

- **H0 and BETA are weakly identified and trade off — the exposure–hazard slope sits on a flat
  collinear ridge.** `H0·exp(BETA·Cc)` is collinear in (H0, BETA), and a single dose gives only
  a narrow concentration range over which to estimate the slope. NONMEM measures the ridge
  directly: corr(H0, BETA) = −0.91 with a singular Hessian along it. The honest reading is that
  the data constrain the exposure–hazard *curve* well but the individual (H0, BETA) pair poorly.

- **Different tools land at different points on the same ridge — this is the expected behaviour,
  not a discrepancy.** NONMEM (LAPLACE, started at the truth-valued initials 0.015 / 0.25)
  stays near H0 ≈ 0.0149, BETA ≈ 0.251; nlmixr2 (FOCEI *and* derivative-free BOBYQA) converges
  to H0 ≈ 0.022, BETA ≈ 0.19. NONMEM's near-truth landing is *start-value dependent on a flat
  ridge* — not independent truth recovery — and nlmixr2's agreement between its own gradient-based
  and derivative-free optimizers confirms its point is a genuine optimum, not a convergence
  artifact. Both are valid optima of an under-determined likelihood.

- **Acceptance check for ferx:** match the identified block (CL, V, KA, ω²(CL), prop. error)
  to the nlmixr2/NONMEM columns within ~2–3 significant figures, and land *somewhere on the
  H0/BETA ridge* (a plausible H0·exp(BETA·Cc) curve) rather than at a specific (H0, BETA) point.
  The anchor validates cross-tool agreement on what is identifiable, plus three-way agreement
  that H0/BETA is a ridge — not exact-truth recovery of H0/BETA.

## To reproduce

```
Rscript simulate.R                       # -> pktte_joint.csv (deterministic, seed 20260628)
Rscript nlmixr2.R                        # nlmixr2 FOCEI / BOBYQA
nmfe76 nonmem.ctl nonmem.lst             # NONMEM (licensed)
cargo run --release --features survival -- pktte_joint_fit.ferx --data pktte_joint.csv
```

Toolchain notes:
- The committed `nonmem.ctl` is the one that ran. It includes two model-neutral NM-TRAN
  corrections — `CC` recomputed in `$ERROR` rather than reusing the `$DES` global `CONC`
  (error 292), and the EPS-containing `Y` flattened out of nested `IF`s (error 326). See
  `nonmem_run_summary.md`.
- If R's configured gfortran is missing, run nlmixr2 with
  `R_MAKEVARS_USER=<Makevars> Rscript nlmixr2.R` where the Makevars sets
  `FLIBS=-L/usr/local/gfortran/lib -lgfortran -lquadmath`.
