# NONMEM run summary — joint PK-TTE (`nonmem.ctl`)

Self-contained record of the NONMEM run of the joint PK-TTE anchor model, for
independent evaluation. Companion files in this folder: `nonmem.ctl` (control
stream, as run), `nonmem.lst` (full output), `nonmem.ext` (iteration/estimates),
`pktte_joint.tab` (tables), `expected.md` (cross-tool comparison table),
`pktte_joint.csv` (data).

## What was run

| | |
|---|---|
| Model | Oral 1-cpt PK (conc on CMT 2) + drug-driven hazard `h = H0·exp(BETA·Cc)`, `Cc = A(2)/V`, cumulative hazard as ODE state CMT 3; event on CMT 3 |
| Estimation | `$ESTIMATION METHOD=COND LAPLACE INTER MAXEVAL=9999` (joint Gaussian PK + F_FLAG likelihood for TTE rows) |
| Covariance | `$COVARIANCE MATRIX=S UNCONDITIONAL` (see note below) |
| Software | NONMEM 7.6.0 (gfortran-9), Docker image `nonmemdocker:V0.1`; license: InsightRX |
| Data | `pktte_joint.csv` — N=120 subjects, 480 normal (PK) obs + TTE rows (78 events / 42 right-censored) |
| Run date | 2026-06-28 |
| Outcome | **MINIMIZATION SUCCESSFUL**; covariance successful via S matrix; 3.6 significant digits in final estimate |

## Control-stream corrections (model-neutral)

The control stream as originally written did not pass NM-TRAN; two fixes were
required to run it on NONMEM. Both preserve the model mathematically (same
predictions, same likelihood) — they only satisfy NM-TRAN's syntax rules.

1. **Error 292 — global variable re-defined.** `CONC = A(2)/V` is defined in
   `$DES` (needed for `DADT(3)`) and was re-defined in `$ERROR`. `$DES` variables
   are global and may not be reassigned in `$ERROR`. Fix: recompute under a new
   name in `$ERROR`, `CC = A(2)/V` (identical value, from the compartment amount
   at the event time).

2. **Error 326 — random variable in a nested IF.** `Y` (which contains `ERR(1)`)
   was assigned inside `IF…THEN…ELSE…ENDIF` with further nested single-line `IF`s.
   NM-TRAN forbids an EPS-containing `Y` inside nested IF structures. Fix:
   flatten `$ERROR` — put the EPS-containing PK prediction at the top level, then
   overwrite `Y` for the TTE rows with flat single-line `IF`s:

   ```
   F_FLAG = 0
   IF (CMT.EQ.3) F_FLAG = 1
   Y = CC*(1 + ERR(1))                          ; Gaussian PK observation
   IF (CMT.EQ.3.AND.DV.EQ.1) Y = HAZ*EXP(-CHZ)  ; exact event:    h(T)*S(T)
   IF (CMT.EQ.3.AND.DV.EQ.0) Y = EXP(-CHZ)       ; right-censored: S(T)
   ```

## Estimates

Simulation truth and nlmixr2 FOCEI shown for reference (full table in
`expected.md`). SEs are from `MATRIX=S` (see covariance note).

| Parameter | Truth | nlmixr2 FOCEI | **NONMEM** | SE | RSE |
|-----------|------:|--------------:|-----------:|-----:|----:|
| CL        | 1.0   | 1.026         | **1.031**    | 0.0291   | 2.8% |
| V         | 10.0  | 9.946         | **9.897**    | 0.115    | 1.2% |
| KA        | 1.0   | 0.9935        | **0.9796**   | 0.0183   | 1.9% |
| H0        | 0.015 | 0.02100       | **0.01491**  | 0.00402  | 27%  |
| BETA      | 0.25  | 0.2014        | **0.2506**   | 0.0455   | 18%  |
| ω²(CL)    | 0.09  | 0.0803        | **0.0814**   | 0.0117   | 14%  |
| σ² (prop) | 0.01  | 0.0105*       | **0.00844**  | 0.000555 | 6.6% |

\* nlmixr2 reports prop. **sd** 0.1023 (≈ var 0.0105); NONMEM σ² = 0.00844 → sd 0.0919.

Diagnostics: ETABAR p = 0.60 (η mean not different from 0); η-shrinkage 4.0% (SD),
ε-shrinkage 11.8% (SD). **corr(H0, BETA) = −0.91.**

## Key interpretation points (for evaluation)

- **PK block agrees across tools.** CL, V, KA, ω²(CL), and the proportional error
  match nlmixr2 to ~2–3 significant figures. This is the valid cross-tool check.

- **H0 / BETA are weakly identified and trade off.** The R matrix (Hessian) is
  algorithmically singular along the (H0, BETA) ridge, so the default sandwich
  covariance aborts; `UNCONDITIONAL` cannot fix a singular matrix. `MATRIX=S`
  (gradient cross-product) does produce SEs, but because R is singular these
  **S-based SEs likely understate the true H0/BETA uncertainty** — treat them as a
  lower bound. The −0.91 correlation and the elevated RSEs (27% / 18% vs 1–3% for
  PK) are the honest signal. A profile likelihood or bootstrap would give truthful
  intervals. NONMEM's H0/BETA landed near their initial values (which equal the
  sim truth, 0.015 / 0.25); on a flat ridge that is start-value dependent, **not**
  independent truth recovery. nlmixr2 (FOCEI and BOBYQA) landed elsewhere on the
  same ridge (H0 ≈ 0.022, BETA ≈ 0.19).

- **OFV is not cross-tool comparable.** NONMEM OFV = 626.65 *without* the
  `N·log(2π) = 882.18` constant (N = 480 normal obs), or 1508.83 *with* it. Neither
  matches nlmixr2's −2LL of −1589.60: the tools normalize the F_FLAG TTE-row
  likelihood differently, and LAPLACE ≠ FOCEI on the non-Gaussian rows. For a
  likelihood-based cross-check, compare **ΔOFV vs Δ(−2LL)** between nested models
  (e.g. BETA fixed to 0) — the tool-specific constants cancel in the difference.

## Reproduce

```bash
# from this folder, using the NONMEM Docker image
docker run --rm -v "$PWD":/work -w /work nonmemdocker:V0.1 \
  /opt/NONMEM/nm760/run/nmfe76 nonmem.ctl nonmem.lst
```
