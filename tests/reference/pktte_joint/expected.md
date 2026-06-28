# Joint PK-TTE anchor — expected values (#564)

Cross-tool reference for ferx's ODE-accumulated joint PK-TTE fit. All tools fit the
**identical** model + dataset (`pktte_joint.csv`): oral 1-cpt PK (concentration on CMT 2)
+ drug-driven hazard `h = H0·exp(BETA·Cc)`, `Cc = central/V`, accumulated as an ODE state,
with the event on CMT 3. N=120, 78 events / 42 right-censored (~35%).

- ferx:    `pktte_joint_fit.ferx`  (FOCEI)
- nlmixr2: `nlmixr2.R`             (FOCEI; cross-checked with BOBYQA outer optimizer)
- NONMEM:  `nonmem.ctl`            (`METHOD=COND LAPLACE INTER`)

## Estimates

| Parameter | Truth | ferx FOCEI | nlmixr2 FOCEI | nlmixr2 BOBYQA | NONMEM LAPLACE |
|-----------|------:|-----------:|--------------:|---------------:|---------------:|
| CL        | 1.0   | _pending_  | 1.026         | 1.029          | _hand-off_     |
| V         | 10.0  | _pending_  | 9.946         | 9.951          | _hand-off_     |
| KA        | 1.0   | _pending_  | 0.9935        | 0.9943         | _hand-off_     |
| H0        | 0.015 | _pending_  | 0.02100       | 0.02206        | _hand-off_     |
| BETA      | 0.25  | _pending_  | 0.2014        | 0.1930         | _hand-off_     |
| ω²(CL)    | 0.09  | _pending_  | 0.0803        | 0.0801         | _hand-off_     |
| prop. sd  | 0.10  | _pending_  | 0.1023        | 0.1023         | _hand-off_     |
| −2LL      | —     | _pending_  | −1589.60      | −1589.68       | _hand-off_     |

(nlmixr2 estimates are on the natural scale: `CL = exp(lcl)`, etc.)

## Notes

- **PK parameters (CL, V, KA), ω²(CL), and the proportional error recover cleanly.**
- **H0 and BETA are weakly identified and trade off.** FOCEI and BOBYQA — gradient-based and
  derivative-free outer optimizers — land at the *same* optimum (H0 ≈ 0.022, BETA ≈ 0.19),
  offset from the simulation truth (0.015 / 0.25). This is a property of the design, not a
  tool/convergence issue: `H0·exp(BETA·Cc)` is collinear in (H0, BETA), and a single dose
  gives a narrow concentration range over which to estimate the exposure–hazard slope. FOCEI's
  first pass reported "false convergence (8)"; BOBYQA converges cleanly to the same place,
  confirming the optimum.
- **The anchor therefore validates cross-tool *agreement* at this optimum**, not exact-truth
  recovery. The acceptance check for ferx is that its FOCEI estimates match the nlmixr2 columns
  to ~2–3 significant figures (and NONMEM once the licensed run is filled in).

## To reproduce

```
Rscript simulate.R                       # -> pktte_joint.csv (deterministic)
Rscript nlmixr2.R                        # nlmixr2 FOCEI/BOBYQA
nmfe75 nonmem.ctl nonmem.lst             # NONMEM (licensed)
cargo run --release --features survival -- pktte_joint_fit.ferx --data pktte_joint.csv
```

Toolchain note: if R's configured gfortran is missing, run nlmixr2 with
`R_MAKEVARS_USER=<Makevars> Rscript nlmixr2.R` where the Makevars sets
`FLIBS=-L/usr/local/gfortran/lib -lgfortran -lquadmath`.
