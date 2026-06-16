# Transit Absorption (Two-Compartment ODE)

This example demonstrates a two-compartment model with transit-compartment absorption and allometric scaling on body weight. The complete model file is [`examples/transit_2cpt.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/transit_2cpt.ferx).

> 💡 **Tip:** this page codes the transit chain as **explicit** ODE states (fixed integer `n`). For most models the built-in [`transit(n, mtt)`](../model-file/absorption.md) input-rate function is simpler — one line, with a single **continuous** (estimable) `n` — see `examples/transit_savic.ferx`.

## When to use

Use a transit-compartment chain when:
- The IWRES vs. time plot shows a systematic under-prediction during the absorption phase (an S-shaped absorption profile)
- A lag-time model fits but the Durbin-Watson statistic is still below 1.5
- The drug has delayed gastric emptying, dissolution-limited absorption, or enterohepatic recirculation

The transit-compartment model (Savic et al., 2007) avoids an arbitrary lag-time parameter by distributing absorption over `n` sequential compartments, each with rate constant `KTR = (n+1)/MTT`, where `MTT` is the mean transit time.

## Dataset

```csv
ID,TIME,DV,EVID,AMT,CMT,MDV,WT
1,0,.,1,250,1,1,95.7
1,0.5,0.0884,0,.,4,0,95.7
...
```

`AMT` is in mg; `DV` (central compartment, CMT=4) is in mg/L. `WT` is body weight in kg, used for allometric scaling.

## Model file

This is the contents of [`examples/transit_2cpt.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/transit_2cpt.ferx):

```
[parameters]
  theta TVCL(5.0,    0.1,  100.0)
  theta TVV1(50.0,   5.0,  500.0)
  theta TVQ(10.0,    0.1,  200.0)
  theta TVV2(100.0,  5.0, 1000.0)
  theta TVMTT(1.0,   0.1,   10.0)

  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_KA ~ 0.09

  sigma PROP_ERR ~ 0.30 (sd)

[individual_parameters]
  CL   = TVCL * (WT / 70)^0.75 * exp(ETA_CL)
  V1   = TVV1 * (WT / 70)^1.00 * exp(ETA_V1)
  Q    = TVQ
  V2   = TVV2 * (WT / 70)^1.00
  KTR  = 4.0 / TVMTT
  KA   = KTR * exp(ETA_KA)

[structural_model]
  ode(obs_cmt=central, states=[transit1, transit2, transit3, central, peripheral])

[odes]
  d/dt(transit1)   = -KTR * transit1
  d/dt(transit2)   =  KTR * transit1 - KTR * transit2
  d/dt(transit3)   =  KTR * transit2 - KTR * transit3
  d/dt(central)    =  KA  * transit3 / V1
                      - (CL / V1 + Q / V1) * central
                      + Q / V2 * peripheral
  d/dt(peripheral) =  Q * central / V1 - Q / V2 * peripheral

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  maxiter    = 500
  covariance = true
```

Key design choices:
- **3 transit compartments** (n=3): `KTR = (3+1)/MTT = 4/MTT`. Increasing `n` steepens the absorption peak but adds no new parameters — only `MTT` is estimated.
- **`obs_cmt=central`**: the ODE state for `central` is concentration (mg/L); incoming flux from transit3 is divided by `V1` in the ODE to keep units consistent.
- **Allometric scaling**: `CL ∝ WT^0.75` and `V ∝ WT^1.0` with reference weight 70 kg.

## Running the fit

```bash
ferx examples/transit_2cpt.ferx --data data/transit_2cpt.csv
```

Or via the Rust API:

```rust
let result = fit_from_files("examples/transit_2cpt.ferx", "data/transit_2cpt.csv")?;
println!("MTT = {:.2} h", result.theta["TVMTT"].estimate);
```

## Interpreting output

Check that the fit reports `converged: true` and that ETA shrinkage for all three random effects is below ~0.7. High shrinkage on `ETA_KA` suggests the absorption phase is not well-characterised in the data — consider fixing `KTR = (n+1)/MTT` to the population value (`KA = KTR`, no IIV on `ETA_KA`).

The Durbin-Watson statistic should be closer to 2 than for a simple one-compartment oral model on the same data.

## Tips

- **Number of transit compartments**: start with n=3 (fixed). Only estimate `n` as a continuous parameter if the OFV improvement justifies it and the data are rich.
- **IIV on KA vs. MTT**: IIV on `KA` in this model scales the entry rate from the last transit compartment; it captures variability in the absorption rate, not in the delay. To model variability in `MTT` (the delay itself), add `omega ETA_MTT` and set `MTT = TVMTT * exp(ETA_MTT)` in `[individual_parameters]`.
- **Allometric scaling**: the reference weight (70 kg) is conventional but can be replaced by the dataset median without changing the exponent. Fixing `Q` and `V2` to population values (no IIV) is common when peripheral data are sparse.
- **FOCEI vs. FOCE**: use FOCEI (`method = focei`) for proportional error models — the interaction term matters most when individual predictions span a large dynamic range (which transit-compartment models tend to produce).
