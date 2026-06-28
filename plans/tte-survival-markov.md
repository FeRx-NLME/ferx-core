# Plan: Non-Gaussian NLME Models — TTE, Survival, RTTE, Markov, and Categorical

**Status:** Phase 1 **and Phase 1b complete** — ferx-core PRs #190, #192, #206 (Phase 1), #441 (validation), #442 (name threading), #494, #501, #526 (Phase 1b competing risks), #563 (#531 cleanup) all merged; ferx-r PRs #134 & #142 merged. **Phase 2 Slice 2.1 (Joint PK-TTE, ODE hazard accumulator, fit path) COMPLETE — #564 via PR #567 (squash `657800ee`) merged 2026-06-28; ferx-r pin bump = draft PR #208.** Open follow-ups: `predict_survival` R wrapper + R-side TTE test; #469 (FOCEI nonlinear-frailty ω² ~17% high vs NONMEM, spin-off of #440, PR #571); #570 (joint-fit double-solve perf — *not urgent*: full 120-subj fit is 5 s in release). **NEXT: Phase 2 Slice 2.2 — drug-driven event-time simulation (`integrate_until_threshold` root-finder + SSE).**  
**Scope:** Active implementation — Phase 2 Slice 2.2 (simulation; Slice 2.1 merged)  
**Revised:** 2026-06-28 (Slice 2.1 #564 merged via #567; three-way anchor ferx≈NONMEM≈nlmixr2 landed + a theta-SE back-transform bug it caught was fixed; ferx-r#208 pin bump kicked off; Slice 2.2 now active; #440 spun off #469; markov/categorical phases untouched)

---

## 0. Executive Summary

The user requested support for TTE/survival models, joint PK-TTE, RTTE, and continuous-time
Markov models (CTMM). Deep research across NONMEM documentation, nlmixr2/saemix/Monolix docs,
tutorial and methods papers, and adjacent fields reveals:

1. **One shared infrastructure primitive** covers all non-Gaussian models: a generalized
   per-subject log-likelihood dispatch. TTE, ordinal, binary, Poisson, NB, CTMM, and HMM all
   plug into the same estimation machinery once this primitive exists.

2. **The NONMEM CTMM dataset approach (EVID=3 + A0_FLG) is architecturally incompatible with
   ferx** (§3.2). The matrix exponential approach is equivalent, cleaner, and correct.

3. **f-SAEM** (Laplace-proposal MH for the SAEM E-step) is a concrete, well-characterized
   improvement to ferx's SAEM implementation that would dramatically accelerate convergence
   for all non-Gaussian models (§9.1).

4. **BHHH (gauss_newton.rs) already extends to non-Gaussian** — the information-matrix
   identity holds for any well-specified probability model (§9.4).

5. **mCTMM (minimal CTMM)** is a single-parameter reduction of CTMM that is identifiable
   under irregular observation times — an ideal stepping stone before full CTMM (§6.3, Phase 4c).

6. **Simulation and prediction are a first-class part of the work, not free.** The current
   `simulate()`/`predict()` path is hardcoded Gaussian (`dv_sim = ipred + σ·ε`, scalar
   `pred`). Each endpoint needs its own sampler and prediction shape; TTE/RTTE/CTMM make the
   observation *times* random outputs, requiring an ODE event-location root-finder and a
   Gillespie engine. This unlocks license-free simulation-estimation (SSE) validation. See
   §8.8; deliverables are threaded into every phase in §12.

---

## 1. Model Taxonomy

### 1.1 Requested model types

| Type | Likelihood | Estimation (recommended) | Priority |
|---|---|---|---|
| Parametric TTE (standalone) | `H(T) − δ·log h(T)` | Laplace (FOCEI) | Phase 1 |
| Joint PK-TTE | Gaussian + TTE mixed | Laplace | Phase 2 |
| Repeated TTE (RTTE) | Σ log h(t_k) − H(T) | **SAEM / IMP** | Phase 3 |
| Time-homogeneous CTMM | Σ log P[s_{k+1}|s_k, Q, Δt] | Laplace or SAEM | Phase 5 |
| Time-inhomogeneous CTMM | same + drug-driven Q(t) | Laplace or SAEM | Phase 6 |

### 1.2 Near-zero-cost extensions (same generalized-LL infrastructure)

| Type | Likelihood | Estimation | Priority |
|---|---|---|---|
| Binary / Bernoulli | `DV·log p + (1-DV)·log(1-p)` | Laplace or SAEM | Phase 4 |
| Ordered categorical (prop. odds) | `log P[Y=k]` cumulative logit | Laplace or SAEM | Phase 4 |
| Poisson count | `-λ + k·log λ − log k!` | Laplace or SAEM | Phase 4 |
| Negative binomial | NegBin(k; r, p) | Laplace or SAEM | Phase 4 |
| DTMM | `log P_{s_k → s_{k+1}}` direct | Laplace or SAEM | Phase 4b |
| mCTMM (minimal CTMM, 1 param) | same as CTMM, single rate q for adjacent states (§3.4) | Laplace or SAEM | Phase 4c |

### 1.3 Longer-term extensions (deferred)

- HMM (hidden Markov models) — forward algorithm O(T·S²), Viterbi for MAP state sequence
- IRT (item response theory) — generalization of proportional odds with item discrimination
- Zero-inflated Poisson / NB
- Beta distribution (proportions on (0,1))
- Frailty / competing risks refinements
- Joint longitudinal-survival with copula or shared random effects
- Cox partial likelihood (semi-parametric, no baseline hazard assumption)
- Full Bayesian (NUTS / Stan comparison)

---

## 2. The Core Infrastructure Primitive

### 2.1 Why one primitive covers all model types

In NONMEM, three values of `F_FLAG` select the likelihood mode:

| F_FLAG | Interpretation of Y |
|---|---|
| 0 | `Y` is a prediction; NONMEM applies Gaussian residual model |
| 1 | `Y` is the **likelihood** of the datum `p(y|η,θ)` |
| 2 | `Y` is **−2 × log-likelihood** |

The pattern for a mixed dataset (continuous PK + TTE events):

```fortran
; In $ERROR
IF (TYPE.EQ.0) THEN
    F_FLAG = 0
    Y = F + F*ERR(1)           ; Normal PK prediction
ELSE
    F_FLAG = 1
    Y = EXP(-CUMHAZ) * HAZNOW  ; TTE likelihood
ENDIF
```

In nlmixr2, `ll(endpoint) ~ expression` replaces `DV ~ distribution(...)`:

```r
ll(tte) ~ event * log(h) - H   ; log-likelihood contribution
```

In saemix (v3.4+), the model type `"likelihood"` accepts a function returning
log-probabilities per observation.

**ferx needs the same abstraction.** Currently `individual_nll` hardcodes
`Σ(y-f)²/V + log V`. The generalization:

```
individual_nll = data_term(η) + eta_prior(η, Ω) + ½ log|Ω|
```

where `data_term(η)` dispatches on endpoint type (§8.5). The eta prior and `½ log|Ω|`
are identical across all model types. SAEM's MH step and IMP automatically benefit
because they only call `individual_nll`.

### 2.2 The NONMEM Laplacian OFV formula — what ferx must compute

NONMEM's Laplacian objective function is the −2·log Laplace-approximated marginal likelihood.
Written out with **every** term (this is what §8.6 implements — keep the two consistent):

```
OFV = 2 × Σᵢ [ −log p(yᵢ|η̂ᵢ)        ; data term: −log L_i at the EBE
              + ½ η̂ᵢᵀ Ω⁻¹ η̂ᵢ        ; eta prior (quadratic form)
              + ½ log|Ω|              ; prior normalizer
              + ½ log|Dᵢ + Ω⁻¹| ]    ; Laplace curvature (the term FOCE drops)
```

The `(n/2)·log 2π` from the prior normalizer and from the Laplace factor `½ log|2πH|`
**cancel**, which is why no 2π appears above — do not double-count them. `H(η̂ᵢ)` is the
**Hessian of the negative log-posterior** at the EBE η̂ᵢ:

```
H(η̂ᵢ) = ∂²[−log p(yᵢ|η) − log p(η|Ω)] / ∂η²  |_{η=η̂ᵢ}
         = D_i^data + Ω⁻¹                              (Dᵢ ≡ ∂²(−data_term)/∂η²)
```

For Gaussian observations, the existing `foce_subject_nll_interaction` computes this
analytically. For non-Gaussian observations, `D_i^data = ∂²(-data_term)/∂η²` must be
computed by finite differences.

**Key difference from standard FOCE**: FOCE drops the `½ log|det(D_i^data)|` contribution
(the data-curvature term). LAPLACIAN retains it. For binary/ordinal/TTE data this term is
non-negligible and its omission causes systematic bias in variance parameters.

### 2.3 Outer Laplace for non-Gaussian (shared FD Hessian)

```rust
fn data_term_hessian_fd(
    eval:    impl Fn(&[f64]) -> f64,  // data_term(eta) = −log p(y|eta)
    eta_hat: &[f64],
    eps:     &[f64],                  // per-dimension step; see §9.3 (Shi 2021) for tuning
) -> DMatrix<f64> {
    let n = eta_hat.len();
    // Evaluate data_term at eta_hat perturbed by `d` (length n).
    let at = |d: &[f64]| {
        let e: Vec<f64> = (0..n).map(|i| eta_hat[i] + d[i]).collect();
        eval(&e)
    };
    let mut h = DMatrix::zeros(n, n);
    for j in 0..n {
        for k in 0..=j {
            // 4-point central stencil for ∂²/∂η_j∂η_k (reduces to the standard
            // second difference with step 2·eps when j == k).
            let mk = |sj: f64, sk: f64| {
                let mut d = vec![0.0; n];
                d[j] += sj * eps[j];
                d[k] += sk * eps[k];
                d
            };
            let entry = (at(&mk(1.0, 1.0)) - at(&mk(1.0, -1.0))
                       - at(&mk(-1.0, 1.0)) + at(&mk(-1.0, -1.0)))
                       / (4.0 * eps[j] * eps[k]);
            h[(j, k)] = entry;
            h[(k, j)] = entry;
        }
    }
    h
}
```

Cost: the 4-point stencil costs 4 evals per (j,k) pair over `n(n+1)/2` pairs ⇒ **2·n(n+1)
evaluations** per subject per outer iteration (n_eta = 4 ⇒ 40). Step size: per-dimension
harmonic mean of gradient norms (Shi 2021, §9.3; used by nlmixr2 generalized FOCEI;
0.8–7.2× slower than standard FOCEI — acceptable).

---

## 3. The Mathematics

### 3.1 TTE individual likelihood

For subject _i_ with event/censoring time T_i and event indicator δ_i ∈ {0,1}:

```
log L_i(η) = δ_i · log h(T_i | η, θ) − H(T_i | η, θ)
```

NONMEM F_FLAG=1 encoding (exact per-observation likelihood, not log-likelihood):

```fortran
; DV=1: exact event        → likelihood = h(T) × S(T) = h × exp(-H)
; DV=0: right censored     → likelihood = S(T) = exp(-H)
; DV=2: interval censored  → likelihood = S(T_L) - S(T_R) = exp(-H_L) - exp(-H_R)
IF (DV.EQ.0) Y = EXP(-CUMHAZ)
IF (DV.EQ.1) Y = EXP(-CUMHAZ) * HAZNOW
IF (DV.EQ.2) Y = EXP(-CUMHAZ_PREV) - EXP(-CUMHAZ)
```

ferx should support all three row types: right-censored (DV=0), exact event (DV=1),
and interval-censored (DV=2). The interval-censored form arises in RTTE when observation
times don't coincide with event times.

Under **left truncation (delayed entry)** — a subject entering the risk set at `T_entry > 0`
— the cumulative hazard term becomes `H(T_i) − H(T_entry,i)` (the likelihood conditions on
survival to entry). See §3.6 for the full treatment; with `T_entry = 0` this reduces to the
form above.

### 3.2 Parametric hazard families

| Name | h(t) | H(t) | Notes |
|---|---|---|---|
| Exponential | λ | λt | Memoryless; standard validation target |
| Weibull | (α/λ)(t/λ)^{α-1} | (t/λ)^α | Most widely used in pharmacometrics |
| Gompertz | λ·exp(γt) | (λ/γ)(e^{γt}−1) | Increasing hazard; used in aging/oncology |
| Log-logistic | αλ^α t^{α-1} / (1+(λt)^α)² | log(1+(λt)^α) | Non-monotone hazard |
| Log-normal | φ-based | Φ-based | Comparable to Weibull, different tail |

Drug effects link to hazard via log-linear or Emax: `h = h₀ · exp(β·C(t))` or
`h = h₀ · (1 − Emax·C / (EC50 + C))`.

#### Proportional hazards (PH) vs. accelerated failure time (AFT)

A covariate can enter a parametric survival model two ways, and the distinction matters for
interpretation and for matching reference software (flexsurv exposes both explicitly):

- **PH:** covariate multiplies the hazard — `h(t|X) = h₀(t)·exp(β'X)`. `exp(β)` is a hazard
  ratio. This is the **default** when the DSL puts the covariate on the hazard/rate (e.g.
  `h = h₀·exp(β·C)` above, or `lambda = THETA_LAMBDA*exp(β·COV)` for Exponential/Gompertz).
- **AFT:** covariate scales time — `h(t|X) = exp(−β'X)·h₀(t·exp(−β'X))`, i.e. the covariate
  acts on the Weibull/log-logistic/log-normal **scale** parameter. `exp(β)` is a time ratio.
  Expressed by putting the covariate on the scale: `lambda = THETA_LAMBDA*exp(β·COV)` with
  the Weibull written in (shape, scale) form.

**Watch the parameterization trap.** In the §3.2 table the Weibull `λ` is the **scale** (a
time: `H = (t/λ)^α`), *not* a rate. So putting a covariate on that `λ` is **AFT**, not PH.
Weibull **PH** multiplies the whole hazard: `h = (α/λ)(t/λ)^{α−1} · exp(β'X)` — implement it as
a separate multiplicative factor (the `loghr` term in §8.4), not by editing `λ`. For the same
data the two fits relate by `β_PH = −α · β_AFT`. By contrast, for **Exponential** (`h = λ`) and
**Gompertz** (`h = λ·e^{γt}`) the `λ` *is* a rate, so a covariate on `λ` genuinely is PH and PH≡AFT
collapses for Exponential. Rule of thumb: covariate **on the hazard** = PH (hazard ratio);
covariate **on the scale/time** = AFT (time ratio). The plan defaults to PH (`loghr`) and
documents the AFT recipe; always report which was used.

### 3.3 RTTE (Repeated TTE)

Multiple events `t₁ < t₂ < … < t_K` within observation window `[0, T]`. There are **two
distinct RTTE models**, and ferx must let the user pick (DSL `clock = forward | reset`,
§8.4). Conflating them is a common and serious modeling error.

**Clock-forward / total time (Andersen–Gill) — DEFAULT.** Hazard `h(t)` is a function of
*absolute* time t (and/or PK). The cumulative hazard accumulates continuously over `[0, T]`
and is **NOT reset** at events:

```
log L_i(η) = Σ_{k=1}^{K_i} log h(t_k|η) − H(T_i|η)        (no reset; H over [0,T])
```

This is what most pharmacometric RTTE (constant or PK-driven hazard) uses.

**Clock-reset / gap time (renewal) — OPTION.** Hazard depends on time *since the previous
event*, `u = t − t_{k−1}`. The accumulator is **reset to 0 at each event**:

```
log L_i(η) = Σ_{k=1}^{K_i} log h(Δ_k|η) − Σ_{k=1}^{K_i+1} H(Δ_k|η)   (Δ_k = inter-event gaps;
                                                                       final gap censored at T)
```

Clock-reset requires the selective per-state ODE reset (§8.8.6); clock-forward does not.
For a time-homogeneous hazard the two coincide; they differ whenever `h` varies with time.

**Estimation recommendation from literature** (Karlsson et al. 2009):

| Method | Event rate <43% (bias on ω²) | Runtime |
|---|---|---|
| Laplace | −91% to −96% (severe) | 0.3 s |
| SAEM | Low (acceptable) | 19 s |
| IMP | Low (acceptable) | 23 s |

**Default for RTTE: SAEM or IMP. Document Laplace as available but warn.**

NONMEM MTIME implementation detail: `MTIME(i)` (i=1..30) defines model event times;
`MNEXT(i)=1` during approach to the time point, `MPAST(i)=1` after it has passed.
The cumulative hazard ODE uses these to accumulate H over each interval. ferx does not
need MTIME — interval boundaries are implicit in the RTTE data records.

### 3.4 CTMM — Critical Design Decision

#### NONMEM's approach: EVID=3 + A0_FLG (NOT feasible in ferx)

NONMEM's `A0_FLG` equals 1 only when `NEWIND ≤ 0` (first record of a subject),
not at every EVID=3 record. For CTMM, the actual mechanism involves Fortran code in
`$PRED` that executes during the record-by-record advance:

```fortran
; At each EVID=3 record (state observation at time t_k):
A_0(1) = (DV.EQ.1)   ; probability = 1 if observed in state 1, else 0
A_0(2) = (DV.EQ.2)   ; probability = 1 if observed in state 2, else 0
```

This sets initial compartment amounts to 0 or 1 based on the **observed DV value**
at that record. The ODE then integrates the Kolmogorov forward equations forward to t_{k+1}.
The likelihood at t_{k+1} is `log A[DV_{k+1}]` — the probability mass in the observed state.

**Why NOT feasible in ferx:**

ferx's EVID=3 handling (`Subject::reset_times`) zeroes ALL compartments at those times —
there is no per-compartment initial value override driven by the observed DV. Adding this
would require:
1. A new field carrying `(time, Vec<f64>)` compartment assignments per CTMM observation
2. The ODE predictor branching on model type to apply data-driven IC instead of zeroing
3. The data reader interpreting `DV` as a state index for these rows (DV has dual meaning)
4. The inner optimizer having access to this during ODE integration

This is a substantial, fragile architectural change. **Verdict: do not use NONMEM EVID=3
approach.**

#### Matrix exponential approach (feasible and recommended)

```
P(Δt) = expm(Q · Δt)      [scaling-and-squaring Padé approximant]
```

Individual log-likelihood:

```
log L_i(η) = Σ_{m=0}^{n_i−1} log [P(Δt_m)]_{s_m, s_{m+1}}
```

Dataset: `(ID, TIME, STATE)` pairs — **no EVID=3 rows**. DV carries integer state index.

For **gradients of P w.r.t. Q entries**, the Van Loan (1978) block-matrix Padé identity
(this is **exact**, not an approximation):

```
∂/∂λ_jk expm(Q) = [expm([[Q, E_jk], [0, Q]])]_{1:S, S+1:2S}
```

where `E_jk` is the matrix with 1 in position (j,k) and 0 elsewhere. One 2S×2S matrix
exponential per parameter gives the exact derivative without finite differences.
For S=3 states: a 6×6 matrix exponential per parameter (fast).

#### mCTMM (minimal CTMM) — highly recommended as stepping stone

Savic & Karlsson (AAPS J, 2017): under the constraint that transition rates between
adjacent states are state-independent, the CTMM reduces to a single parameter model:
`q_jk = q` (constant) for all adjacent (j,k) pairs. The mean equilibration time is
`τ = 1/q`. Steady-state probabilities follow a proportional odds model.

Advantages:
- Identifiable even with irregular observation times
- Natural stepping stone before full CTMM
- Implemented in `msm` package as a special case
- Useful for disease progression models with monotone (worsening) states

### 3.5 Categorical and Count Models

**Binary / Bernoulli:** logit link, `p = logistic(intercept + β·C + ETA)`

```
data_term = -Σ [DV·log(p_k) + (1-DV)·log(1-p_k)]
```

**Ordered categorical (proportional odds):** K categories, K−1 cut-points α_k

```
P[Y ≤ k] = logistic(α_k − β·C − ETA)
P[Y = k] = P[Y ≤ k] − P[Y ≤ k−1]
data_term = -Σ log P[Y = DV_k]
```

**Identifiability — enforce monotone cut-points.** The α_k must be strictly increasing
(α_1 < α_2 < … < α_{K−1}) or `P[Y=k]` can go negative and the model is non-identified
(§17). Implement via an unconstrained parameterization: `α_1 = θ_1`,
`α_k = α_{k−1} + exp(δ_k)` for k ≥ 2. The sign convention here (`−β·C`) means a higher linear
predictor shifts mass to **higher** categories; the §8.4 DSL must use the same sign.

Monolix syntax for reference:
```
logit(P(level<=0)) = th1
logit(P(level<=1)) = th1 + th2
logit(P(level<=2)) = th1 + th2 + th3
```

**Poisson count:** `λ = baseline_rate · exp(β·C + ETA)`

```
data_term = Σ [λ_k − DV_k·log(λ_k) + ln_gamma(DV_k + 1)]
```

Use `ln_gamma(k+1)`, not `log(k!)` (factorial overflows for modest counts). The
`ln_gamma` term is constant in the parameters so it does not affect estimates, but it
**must** be included to match a reference OFV that includes it (NONMEM's F_FLAG Poisson does).

**Negative binomial:** extends Poisson with overdispersion parameter r

```
log p(k; r, μ) = log Γ(k+r) - log Γ(r) - log k! + r·log(r/(r+μ)) + k·log(μ/(r+μ))
```

**DTMM (Discrete-Time Markov):**

Transition probabilities `P_{jk}` are model parameters. At each observation:

```
data_term += -log P[s_k | s_{k-1}, θ, η]
```

Monolix uses cumulative logit: `logit(P(State<=k|State_p=j)) = α_jk + β·C`

**HMM (hidden states, deferred):** Forward algorithm:

```
α₁(s) = π(s) × p(y₁|s)
αₜ(s) = [Σₛ' αₜ₋₁(s') × P(s'→s)] × p(yₜ|s)
L = Σₛ αₜ(s)                    ; total likelihood via log-sum-exp
```

Complexity: O(T·S²) per subject. State estimation via Viterbi (MAP) is a post-hoc
operation on the posterior sequence.

### 3.6 Censoring — Complete Treatment

Censoring is the mechanism by which incomplete event-time information enters the
likelihood correctly. Getting it wrong — or ignoring it — biases all parameter
estimates. Every ferx TTE/RTTE model must handle at minimum right censoring and
exact events; interval censoring is important for periodic-assessment data.

#### Censoring types, likelihoods, and DV coding

| Type | Situation | DV | Likelihood contribution | Ferx priority |
|---|---|---|---|---|
| Exact event | Event observed at known time T | 1 | h(T)·exp(−H(T)) | Phase 1 — required |
| Right censored | Study ends or subject drops out before event | 0 | exp(−H(T)) | Phase 1 — required |
| Interval censored | Event in (T_L, T_R); periodic assessments | 2 | exp(−H(T_L)) − exp(−H(T_R)) | Phase 1 — support |
| Left censored | Event before first observation (extremely rare in PK/PD TTE) | 3 | 1 − exp(−H(T)) | Defer indefinitely |
| Informative | Dropout related to event risk | — | Violates ignorability; needs joint model | Defer |

All standard TTE analyses assume **non-informative censoring**: the censoring mechanism is
independent of the future event probability given the observed history. This is an
untestable assumption that should be documented in model descriptions.

#### NONMEM likelihood code for all three active types

```fortran
$DES
  DADT(1) = HAZNOW           ; CHZ: cumulative hazard ODE state

$ERROR
  CHZ    = A(1)              ; H(T) at current TIME
  IF (DV.EQ.0) THEN          ; right-censored at TIME
    F_FLAG = 1
    Y = EXP(-CHZ)
  ELSE IF (DV.EQ.1) THEN     ; exact event at TIME
    F_FLAG = 1
    Y = HAZNOW * EXP(-CHZ)
  ELSE IF (DV.EQ.2) THEN     ; interval-censored: event in (CHLAST_time, TIME)
    F_FLAG = 1
    Y = EXP(-CHLAST) - EXP(-CHZ)
  END IF
  CHLAST = CHZ               ; carry H(T) into next record (for interval censoring)
```

NONMEM's `CHLAST` is a user-defined variable that persists across records within a
subject. The data has two consecutive rows for each interval-censored observation:
DV=0 at T_L (saves H(T_L) into CHLAST), then DV=2 at T_R.

**ferx approach**: store both bounds directly in `ObsRecord::Event`:

```rust
EventType::IntervalCensored { left: f64, right: f64 }
// data_term = -( exp(-H(left)) - exp(-H(right)) ).ln()
```

No CHLAST needed — both times are available at once from the record.

#### Dataset encoding — one record per subject (simple TTE)

```csv
ID,TIME,DV,CMT,EVID,MDV
1,24.0,0,2,0,0     ; right-censored at t=24
2,15.3,1,2,0,0     ; exact event at t=15.3
3,10.0,0,2,0,0     ; interval-censored: left bound (H saved internally)
3,18.0,2,2,0,0     ;   right bound: event occurred between t=10 and t=18
```

The CMT column routes all rows to the TTE endpoint; DV encodes the censoring type.

#### RTTE — the specific data pattern

RTTE data has **multiple DV=1 rows** per subject plus **one mandatory final DV=0 row**
marking the observation window end T. Without the final DV=0 row the cumulative hazard
H(T) term is undefined.

```csv
ID,TIME,DV,CMT,EVID,MDV
1,5.2, 1,2,0,0     ; 1st event
1,11.7,1,2,0,0     ; 2nd event
1,24.0,0,2,0,0     ; ← mandatory: defines observation window end T=24
2,24.0,0,2,0,0     ; subject 2: no events, fully right-censored
```

The likelihood for subject 1, **clock-forward (default)**: `log h(5.2) + log h(11.7) − H(24.0)`,
where H(24.0) integrates continuously from 0 — NOT reset at event times. Under **clock-reset**
(`clock = reset`, §3.3) the same data instead yields `log h(5.2) + log h(6.5) − [H(5.2) +
H(6.5) + H(12.3)]` on the inter-event gaps (5.2, 11.7−5.2, 24−11.7), with the accumulator
restarted at each event (§8.8.6). The data format is identical; only the `clock` setting differs.

**Interval-censored RTTE events.** When an event is known only to fall in `(t_L, t_R)` (DV=2,
e.g. found at a scheduled visit), it contributes the interval probability `S(t_L) − S(t_R)`
rather than `h·S`, exactly as for single TTE (§3.6 table). Under **clock-forward** the bounds
are absolute times and `S` uses the continuous cumulative hazard:
`exp(−H(t_L)) − exp(−H(t_R))`. Under **clock-reset** the bounds are measured from the previous
event origin (`Δ_L = t_L − t_{prev}`, `Δ_R = t_R − t_{prev}`). Because the *exact* event time is
then unknown, the next gap's renewal origin is ambiguous; the convention is to restart the clock
at `t_R` (the conservative, observed bound) and document it. Interval-censored RTTE is a Phase 3
sub-case, not Phase 1.

#### Competing risks — per-event-type censoring

Each event type occupies its own CMT. A subject experiencing event type A is
right-censored for event type B at the same time:

```csv
ID,TIME,DV,CMT,EVID,MDV
1,15.0,1,2,0,0     ; response event (CMT=2)
1,15.0,0,3,0,0     ; censored for dropout at same time (CMT=3)
```

Each CMT is fitted with its own `HazardSpec`; the shared random effects (eta on PK
parameters) link the two hazard models. This is cause-specific hazard — the standard
pharmacometric approach for competing risks. Implementation is scheduled as **Phase 1b**
(§12): no new infrastructure beyond per-CMT `[event_model]` blocks and the censoring
routing already in Phase 1.

#### Left truncation (delayed entry) — distinct from left *censoring*

A subject is **left-truncated** when it enters the risk set at an entry time `T_entry > 0`
(staggered enrollment, age as the time scale, landmark analysis). Such a subject is in the
study *only because* it had not yet had the event at `T_entry`, so the likelihood must
condition on survival to entry:

```
L_i = h(T_i)^{δ_i} · S(T_i) / S(T_entry,i)
    = h(T_i)^{δ_i} · exp( −[ H(T_i) − H(T_entry,i) ] )
```

Ignoring truncation biases the hazard upward — the event-free pre-entry period is wrongly
counted as at-risk from t=0. This is a **core survival-analysis requirement, not an edge
case**, and is separate from left *censoring* (which is deferred, §3.6 table).

**Data coding:** an optional `TENTRY` column (default 0). When present, the cumulative-hazard
contribution is `H(T) − H(TENTRY)`.

```csv
ID,TIME,TENTRY,DV,CMT,EVID,MDV
1,40.0,30.0,1,2,0,0   ; entered risk set at age 30, event at age 40
2,55.0,35.0,0,2,0,0   ; entered at 35, right-censored at 55
```

**ferx approach:** read `TENTRY` into `ObsRecord::Event.entry_time` (§8.1); in `tte_data_term`,
subtract `H(TENTRY)` (analytic families) or initialize the CHZ ODE integration at `TENTRY`
rather than 0 (ODE-accumulated hazard). For RTTE the same correction applies to the first
at-risk interval. No change to the inner/outer optimizer. **Simulation** mirrors this:
draw conditionally on survival past `T_entry` (i.e. solve `H(T) − H(T_entry) = −log u`).

#### What ferx must enforce

- Every TTE subject must have exactly one terminal row (DV=0 or DV=1) — the last
  observation defines the observation window.
- For RTTE, the final DV=0 row is mandatory even if events were observed; the data
  reader should error if a subject's RTTE records lack a terminal DV=0 row.
- Interval censoring (DV=2) must appear after a preceding DV=0 row for the same
  subject on the same CMT within the same window (this defines T_L).
- If `TENTRY` is present it must satisfy `0 ≤ TENTRY < TIME` for every event row; the data
  reader should error otherwise.

---

## 4. Reference Implementation Details

### 4.1 NONMEM — Exact Syntax

#### F_FLAG / LIKE mechanism

```fortran
$ERROR
IF (TYPE.EQ.0) THEN   ; TYPE: user-defined column distinguishing obs types
    F_FLAG = 0
    Y = F + F*ERR(1)
ELSE IF (TYPE.EQ.1) THEN   ; TTE observation
    CHZ = A(3)              ; cumulative hazard from ODE compartment 3
    HAZNOW = THETA(3) * EXP(-THETA(4) * F)  ; f(concentration F)
    IF (DV.EQ.1) THEN
        F_FLAG = 1
        Y = HAZNOW * EXP(-CHZ)
    ELSE
        F_FLAG = 1
        Y = EXP(-CHZ)
    END IF
END IF

$ESTIMATION METHOD=CONDITIONAL LAPLACIAN INTERACTION
```

#### Estimation block options for non-Gaussian

```
$ESTIMATION METHOD=LAPLACIAN INTERACTION LAPLACE NOABORT PRINT=5
$ESTIMATION METHOD=IMP LAPLACE INTERACTION AUTO=1 ISAMPLE=300 PRINT=5
$ESTIMATION METHOD=SAEM INTERACTION AUTO=1 ISAMPLE=10 PRINT=5 RANMETHOD=S2
$ESTIMATION METHOD=NUTS NBURN=1000 NITER=2000   ; Full Bayesian (NONMEM 7.4+)
```

ISAMPLE guidelines: 10 for SAEM with non-Gaussian; ≥300 for IMP with non-Gaussian.
AUTO=1 enables automatic step-size adaptation.

#### RTTE with MTIME

```fortran
$PK
  MTIME(1) = 5    ; interval boundaries
  MTIME(2) = 10

$DES
  DADT(1) = THETA(1) * EXP(ETA(1))  ; cumulative hazard ODE

$ERROR
  F_FLAG = 1
  IF (DV.EQ.0) Y = EXP(-A(1))       ; censored
  IF (DV.EQ.1) Y = THETA(1)*EXP(ETA(1)) * EXP(-A(1))  ; event
```

#### PRIOR and full Bayesian for TTE

```
$THETAP      ; means for theta priors
  (0,0.1)    ; prior mean for log(lambda)
$THETAPV     ; prior variances
  (1,1)
$OMEGAP      ; prior on omega
  0.25
$OMEGAPD     ; prior degrees of freedom (Inv-Wishart)
  3
$ESTIMATION METHOD=NUTS NBURN=500 NITER=1000
```

LKJ priors available for correlations (`OLKJDF`), half-t for variances (`OVARF`) in
NONMEM 7.5+.

### 4.2 nlmixr2 / rxode2 — Exact Syntax

#### All supported distributions (rxode2ll, 15 total)

| Function | Distribution | Typical use |
|---|---|---|
| `llikNorm(x, mean, sd)` | Normal | Gaussian continuous; same as standard |
| `llikPois(x, lambda)` | Poisson | Count data |
| `llikBinom(x, size, prob)` | Binomial | Binary (size=1) or count-of-successes |
| `llikNbinom(x, size, prob)` | Negative binomial (prob) | Overdispersed count |
| `llikNbinomMu(x, size, mu)` | Negative binomial (mean) | Overdispersed count |
| `llikBeta(x, shape1, shape2)` | Beta | Proportions on (0,1) |
| `llikT(x, df, mu, sigma)` | Student-t | Heavy-tailed residuals |
| `llikWeibull(x, shape, scale)` | Weibull | TTE parametric |
| `llikGamma(x, shape, rate)` | Gamma | PK AUC, positive continuous |
| `llikExp(x, rate)` | Exponential | TTE constant hazard |
| `llikGeom(x, prob)` | Geometric | Count until first success |
| `llikCauchy(x, location, scale)` | Cauchy | Heavy-tailed, outlier-robust |
| `llikUnif(x, min, max)` | Uniform | Rarely used in NLME |
| `llikChisq(x, df)` | Chi-squared | — |
| `llikF(x, df1, df2)` | F | — |

These use Stan C++ log-likelihood implementations internally and support AD.

#### Ordinal model syntax

```r
# Form 1: probability vector (implied last category = 1 - sum)
err ~ c(p0, p1, p2)

# Form 2: named categories
err ~ c(p0=0, p1=1, p2=2, 3)

# In model block, compute category probabilities from logistic:
model({
  lp1 <- alpha1 + beta * Cc
  lp2 <- alpha1 + alpha2 + beta * Cc
  p1 <- exp(lp1) / (1 + exp(lp1))
  p2 <- exp(lp2) / (1 + exp(lp2))
  err ~ c(p1 - 0, p2 - p1, 1 - p2)   ; 3 categories
})
```

#### TTE syntax (ll interface)

```r
gompertz_model <- function() {
  ini({
    log_alpha   <- log(0.003)
    log_gamma   <- log(0.003)
    log_hr      <- -0.5
  })
  model({
    alpha <- exp(log_alpha)
    gamma <- exp(log_gamma)
    h     <- alpha * exp(gamma * time) * exp(log_hr * trt)
    H     <- (alpha/gamma) * (exp(gamma*time) - 1) * exp(log_hr * trt)
    ll(tte) ~ event * log(h) - H
  })
}
fit <- nlmixr(gompertz_model, data, est = "bobyqa")
```

For joint PK-TTE with rxode2 ODE:

```r
pktte_model <- function() {
  ini({
    lCL   <- log(4);   lV  <- log(70)
    lH0   <- log(0.01); lBeta <- -0.5
    eta.CL ~ 0.09; eta.V ~ 0.04
  })
  model({
    CL <- exp(lCL + eta.CL)
    V  <- exp(lV + eta.V)
    k  <- CL / V
    Cc <- center / V                          ; rxode2 ODE compartment
    h  <- exp(lH0 + lBeta * Cc)
    d/dt(depot)  <- -ka * depot
    d/dt(center) <- ka * depot - k * center
    d/dt(cumhaz) <- h                         ; hazard accumulator ODE state
    DV ~ add(sigma) | (cmt == 1)             ; PK observation
    ll(tte) ~ event * log(h) - cumhaz | (cmt == 2)  ; TTE observation
  })
}
```

#### Generalized FOCEI (llik-focei)

The generalized FOCEI replaces the analytic Gaussian data Hessian with a numerical FD
Hessian. Objective function:

```
lᵢ = -Σⱼ llikⱼ - ½ ηᵢᵀ Ω⁻¹ ηᵢ - ½ log(det(2πΩ))
```

Step-size for FD Hessian: Shi (2021) algorithm — harmonic mean of gradient norms at
each observation. Performance: 0.8–7.2× slower than standard FOCEI; acceptable.
Cannot compare OFVs between generalized-FOCEI and standard-FOCEI models.

#### AGQ (Adaptive Gauss-Hermite Quadrature)

```r
fit <- nlmixr(model, data, est = "focei", control = list(nAGQ = 3))
```

`nAGQ=1` = Laplace approximation. For `nAGQ` quadrature nodes and `q` random effects:
`nAGQ^q` evaluation points per subject. Practical for q ≤ 3–5 only.

### 4.3 Monolix — Complete Syntax

#### TTE observation type (Mlxtran)

```
ObservationName = {
    type        = event,
    eventType   = exact,                ; or intervalCensored
    maxEventNumber = 1,                 ; omit for unlimited (RTTE)
    hazard      = h,                    ; expression referencing model vars
    rightCensoringTime = tmax           ; simulation only
}
```

Data: `Y=1` for exact event; `Y=0` for right-censored. A record at t=0 anchors hazard
integration. For RTTE: set `maxEventNumber > 1`; Monolix computes separate VPC per event.

#### Discrete-time Markov (Mlxtran)

```
State = {
    type       = categorical,
    categories = {1, 2, 3},
    dependence = Markov,
    P(State_1=1) = a1,                        ; initial state
    logit(P(State<=1|State_p=1)) = a11,
    logit(P(State<=2|State_p=1)) = a11+a12,
    logit(P(State<=1|State_p=2)) = a21,
    logit(P(State<=2|State_p=2)) = a21+a22
}
```

#### Continuous-time Markov (Mlxtran)

```
State = {
    type       = categorical,
    categories = {1, 2},
    dependence = Markov,
    transitionRate(1, 2) = q12,
    transitionRate(2, 1) = q21
}
```

Monolix integrates Kolmogorov forward equations internally using the matrix exponential.
The user only specifies transition rates. This is the target behaviour for ferx — cleaner
than NONMEM's EVID=3 approach.

#### Ordinal model (Mlxtran)

```
level = {
    type       = categorical,
    categories = {0, 1, 2, 3},
    logit(P(level<=0)) = th1,
    logit(P(level<=1)) = th1 + th2,
    logit(P(level<=2)) = th1 + th2 + th3
}
```

#### SAEM phase structure and tuning

Three phases:
1. **Burn-in**: 5 iterations (default); pure MCMC, no parameter updates; establishes
   MCMC mixing before stochastic approximation starts.
2. **Exploratory**: 150–500 iterations; step-size exponent = 0 (all iterations equal
   weight); simulated annealing constrains variance decreases to ≤5%/iteration to prevent
   premature convergence.
3. **Smoothing**: 50–200 iterations; step-size exponent = 0.7 (**must be >0.5** for
   stochastic approximation to converge almost surely to the MLE).

For non-Gaussian models: the MH proposal in Monolix uses an **independent Gaussian** (not
random-walk) centered at the current EBE with covariance from the Laplace approximation
of the individual log-posterior. This is the f-SAEM approach (§9.1) — no user tuning
required.

### 4.4 saemix (R package, v3.4+)

Non-Gaussian model specification — returns log-probabilities:

```r
tte_model <- function(psi, id, x) {
    T     <- x[, 1];   cens <- x[, 2]   # event time, censoring indicator
    shape <- psi[id, 1];  scale <- psi[id, 2]
    hazard <- (shape/scale) * (T/scale)^(shape-1)
    H      <- (T/scale)^shape
    logpdf <- ifelse(cens == 1, -H, -H + log(hazard))
    return(logpdf)
}

tte_saemix_model <- saemixModel(
    model     = tte_model,
    psi0      = matrix(c(2, 0.5), ncol=2),
    type      = "likelihood",                  ; <-- non-Gaussian flag
    transform.par = c(1, 1)                   ; log-transform both params
)
```

The March 2026 arXiv paper (2603.03154) adds: bootstrap uncertainty quantification for
non-Gaussian models; covariate/variability model selection tools for categorical and
survival data.

---

## 5. Gap Analysis — What ferx-core Currently Lacks

### 5.1 Observation type system ✅ *Resolved — PRs #190, #192*

`ObsRecord::Event`, `EventType`, and `Subject.obs_records` added (behind
`#[cfg(feature = "survival")]`). `CompiledModel.endpoints: HashMap<usize,
EndpointLikelihood>` populated by `[event_model]` parser (PR #192). Binary/ordinal/count
variants deferred to Phase 4.

### 5.2 Individual NLL dispatch ✅ *Resolved — PRs #192, #206*

`individual_nll_into_with_schedule` dispatches through `model.endpoints`; TTE data term
added (2× scaling, FOCEI halving convention). `obs_nll_subject_into` (SAEM M-step) adds
TTE term for each `EndpointLikelihood::Tte` endpoint (PR #206). IOV + TTE joint path
also resolved in PR #206.

### 5.3 Outer Laplace data Hessian for non-Gaussian ✅ *Resolved — PRs #190, #192*

`data_term_hessian_fd` (4-point central stencil) and `shi_step_sizes` (Shi 2021 §3.4)
in `src/survival/mod.rs`. Wired into `foce_subject_nll_interaction_with_tte` (PR #192):
FD Hessian + `½ log|det H_total|` for TTE CMTs; combined with Gaussian Almquist correction.

### 5.4 Model DSL blocks ✅ *Resolved — PR #192, extended PR #206*

`[event_model]` parser landed in PR #192 (Exponential, Weibull, Gompertz; named blocks
for competing risks; `loghr` PH term). PR #206: `[structural_model]`, `[error_model]`,
`[individual_parameters]` all optional when `[event_model]` is the sole endpoint.
Binary/ordinal/count/Markov rate-matrix blocks deferred to Phase 4+.

### 5.5 Cumulative hazard integration *(open — Phase 2)*

Must add CHZ as an extra ODE state for ODE-linked hazard (joint PK-TTE). The existing
RK45 handles augmented states; wiring is Phase 2 work.

### 5.6 Matrix exponential *(open — Phase 5)*

No expm implementation. Need Padé approximant for CTMM. Van Loan trick for gradients.

### 5.7 Data reader extensions ✅ *Resolved — PR #192*

DV=0/1/2 routed to `subject.obs_records` for TTE CMTs; `TENTRY` column auto-detected
for left truncation; non-integer DV on TTE CMT → hard error. State-index and count
routing deferred to Phase 4.

### 5.8 SAEM sigma update for non-Gaussian ✅ *Resolved — PR #206*

SAEM M-step now adds the TTE data term for TTE subjects. Sigma update is Gaussian-only
(no sigma in TTE models), which is correct — skipped automatically when no sigma declared.

### 5.9 f-SAEM proposal *(open — improvement opportunity)*

Current SAEM uses a random-walk MH proposal. Replacing it with a Laplace-based independent
proposal (f-SAEM) would accelerate convergence for all non-Gaussian models (§9.1); this is
Phase 3b and remains **open**. A related but distinct increment has landed: PR #265 ("SAEM
conditional-distribution pass", merged 2026-06-21) characterises each subject's post-fit
`p(η|y)` — conditional mean/SD/draws, the saemix `conddist` / Monolix analog — by
*accumulating* the existing random-walk MH draws after the fit. It does **not** change the
E-step proposal, so it does not implement f-SAEM; but its MH kernels (`mh_steps`,
`mh_steps_componentwise`, `mh_kappa_steps`, now crate-visible in
`src/estimation/saem_conddist.rs`) are reusable as Phase 3b scaffolding.

---

## 6. Feasibility Assessment

### 6.1 Generalized log-likelihood

**High.** Data term dispatch, FD Hessian, endpoint-type routing — all self-contained.
SAEM and IMP extend automatically. GN (Gaussian residual specific) does not extend.
BHHH already works: information-matrix identity `E[∇l∇lᵀ] = -E[∇²l]` holds for any
well-specified model (§9.4).

### 6.2 TTE / RTTE

**High.** ODE CHZ state reuses existing RK45. Inner BFGS finds EBEs of TTE NLL without
change (TTE posteriors are typically unimodal). Nelder-Mead fallback in `inner_optimizer.rs`
provides robustness for atypical surfaces.

### 6.3 CTMM — NONMEM EVID=3 approach NOT feasible; matrix expm IS feasible

See §3.4. Summary:

| Approach | Feasibility | Effort | Notes |
|---|---|---|---|
| NONMEM EVID=3 + A0_FLG | **Not feasible** | Very high | Data-driven ODE IC; dual-meaning EVID=3 |
| Matrix exponential (Padé) | **Feasible** | Medium | ~150 lines nalgebra; same likelihood |
| mCTMM (1-parameter reduction) | **Feasible** | Low | Stepping stone; identifiable |
| Time-inhomogeneous (matrix ODE) | Feasible | Medium | n_states² ODE; existing RK45 |

### 6.4 Categorical / count

**High.** Binary, ordinal, Poisson, NB all map cleanly to generalized data term. Main
work is DSL, data reader for integer DV, and the distribution-specific term functions.

### 6.5 DTMM

**High, simpler than CTMM.** Transition probabilities directly parameterized. Dataset
format: `(ID, TIME, STATE)` pairs with fixed-length intervals. Validation vs. NONMEM
(Bergstrand 2025 supplementary code).

### 6.6 HMM (deferred)

**Medium.** Forward algorithm O(T·S²) is straightforward. Main challenge: HMM requires
marginalization over hidden states per observation — incompatible with single EBE computation.
Requires special handling in inner optimizer (EM over hidden states, not BFGS over η).

---

## 7. Key Literature

### 7.1 TTE / survival

- **Holford NH (2013).** "A time to event tutorial for pharmacometricians." CPT:PSP.
  — Standard pedagogical reference; NONMEM CHZ ODE code; Exponential, Weibull, Gompertz.
- **Lindauer A et al. (2010).** Tumor-growth + TTE joint model; J Clin Oncol.
- **Holford NH et al. (2006).** Disease progression + TTE (levodopa, Parkinson); Mov Disord.
- **flexsurv R package**: Royston P (2015). J Stat Softw. — Parametric TTE with AD gradients.

### 7.2 RTTE

- **Karlsson KE et al. (2009, PAGE abs.).** SAEM vs. FOCE vs. IMP for RTTE; key quantitative
  results (§3.3 above).
- **Holford NH (2013)** also covers RTTE likelihood derivation.
- **Plan EL (2014).** Modeling and simulation of count data. CPT:PSP.

### 7.3 CTMM and Markov

- **Ooi EHS, Plan EL, Bergstrand M (2025).** "Practical guidance for Markov models in drug
  development." CPT:PSP 14:197–216. — DTMM, mCTMM, HMM, IRT+Markov; annotated NONMEM code
  in supplementary.
- **Savic RM, Karlsson MO (2017, AAPS J).** mCTMM (minimal CTMM) parameterization.
- **Jackson CH (2011).** msm package; J Stat Softw. — Matrix exponential CTMM; CAV dataset.
- **Sctmm paper (PMC11247187).** Scalable CTMM via SGD + block-Padé; 13,320 MS patients.

### 7.4 Categorical and count

- **Lacroix BD et al. (2009).** Ordered 5-category PD NONMEM model; J Pharmacol Exp Ther.
- **Plan EL (2011, Uppsala PhD).** PS, PMAK, PMIX, ZIP, GP, NB count models in NONMEM.
  LAPLACE bias: 1.02% average for fixed effects across all count models.
- **Comets E et al. (2026, arXiv:2603.03154).** saemix non-Gaussian; binary (toenail),
  RTTE; bootstrap uncertainty.
- **Jonsson EN, Karlsson MO (1999).** Xpose; ordinal and binary VPC diagnostics.

### 7.5 Methods improvements

- **Shi J (2021).** FD step-size for non-Gaussian FOCEI Hessian; harmonic mean algorithm.
  (Referenced in nlmixr2 generalized FOCEI implementation.)
- **PMC11577698 (2024).** Dynamic survival ODE (dH/dt = h in ODE system); AD-friendly.
- **PMC7373158.** Saddle-point reset for non-Gaussian inner optimization.
- **arXiv:2605.20345 (2025).** CILA (corrected integrated Laplace) via importance sampling.
- **arXiv:2601.17400 (2025).** VAE-NLME; amortized inference; GPU-enabled.
- **Van Loan CF (1978).** Block-matrix Padé for matrix exponential derivatives.

### 7.6 Freely available reference materials — what can be used directly

#### CAV dataset (CTMM — best available, complete output)

Available immediately via `data(cav, package = "msm")`. 622 heart transplant
recipients, 4 states (no CAV / mild / severe / dead), 2846 rows.

**Complete reference output** from two independent implementations (msm package and
mrgsolve blog) that agree exactly:

| Transition | Rate |
|---|---|
| State 1→2 | 0.1279 |
| State 2→1 | 0.2251 |
| State 2→3 | 0.3426 |
| State 3→2 | 0.1306 |
| State 1→4 | 0.0425 |
| State 2→4 | 0.0403 |
| State 3→4 | 0.3065 |

**OFV: 3968.798** (both msm and mrgsolve agree exactly).

Code: `msm::msm()` (R package) and mrgsolve blog (`mrgsolve.org/blog/posts/msm.html`).
The mrgsolve version uses an ODE-based approach structurally similar to ferx — making
it a direct code-level comparison, not just a numerical one.

#### nlmixr2 TTE blog — simulated dataset with reference output

Blog post `blog.nlmixr2.org/blog/2026-05-28-survival-nlmixr2/` has complete inline
R code (dataset simulated from a Gompertz model, 300 patients, 2-arm RCT, fixed seed).

**Reference output (Gompertz model, bobyqa, no random effects):**

| Parameter | Estimate | SE |
|---|---|---|
| log_alpha | −6.173 | 0.295 |
| log_gamma | −5.321 | 0.274 |
| test_log_hr | −0.799 | 0.274 |

−2LL = 2955.64; AIC = 2961.64.
Exponential (nested): −2LL = 3008.71; ΔAIC = 51.

Censoring coded as `event = 0` (censored) / `event = 1` (observed); dataset
reproducible with a fixed seed directly from the blog post.

#### Toenail dataset (binary model)

Available via `data(toenail, package = "saemix")`. 294 patients, binary repeated
outcome. Fitted estimates in Comets et al. (arXiv:2603.03154) using saemix SAEM.

### 7.7 Supplementary materials we must supply

The following do not exist anywhere and must be created by the ferx team
before or alongside each implementation phase.

#### Must create before Phase 1 (TTE)

| File | What it is | How |
|---|---|---|
| `tests/reference/tte_exponential/simulate.R` | Fixed-seed R script → 100-subject exponential TTE dataset | Write now |
| `tests/reference/tte_exponential/tte_exp.csv` | The dataset (committed) | Run simulate.R |
| `tests/reference/tte_exponential/nonmem.ctl` | NONMEM: `$PRED`, F_FLAG=1, Exponential, LAPLACIAN | Write now |
| `tests/reference/tte_exponential/nonmem.lst` | NONMEM output: OFV, theta, omega, SE | Requested during dev — license required, **non-blocking** (added when it arrives; free refs + SSE carry validation meanwhile) |
| `tests/reference/tte_exponential/nlmixr2.R` | nlmixr2 `ll()` Exponential fit on same dataset | Write now |
| `tests/reference/tte_exponential/expected.md` | Reference estimates table (fill after running) | After NONMEM + nlmixr2 |

Same set for **TTE Weibull** and **joint PK-TTE** (Phase 2).

#### Must create before Phase 3 (RTTE)

| File | What it is |
|---|---|
| `tests/reference/rtte_exponential/simulate.R` | 100 subjects; constant-hazard RTTE; ~3 events/subject expected |
| `tests/reference/rtte_exponential/rtte_exp.csv` | Dataset (committed) |
| `tests/reference/rtte_exponential/nonmem.ctl` | NONMEM F_FLAG=1 + MTIME interval pattern |
| `tests/reference/rtte_exponential/nlmixr2.R` | nlmixr2 SAEM (primary reference) |
| `tests/reference/rtte_exponential/expected.md` | Reference estimates |

#### Must create before Phase 4 (categorical / count)

| File | What it is |
|---|---|
| `tests/reference/ordinal_simulated/simulate.R` | 200 subjects; 4-category proportional odds; drug effect |
| `tests/reference/ordinal_simulated/nonmem.ctl` | NONMEM F_FLAG=1, cumulative logit |
| `tests/reference/ordinal_simulated/nlmixr2.R` | nlmixr2 ordinal syntax |
| `tests/reference/binary_toenail/get_data.R` | `data(toenail, package="saemix")` → write to CSV |
| `tests/reference/binary_toenail/nonmem.ctl` | NONMEM F_FLAG=1, logit link |
| `tests/reference/binary_toenail/saemix.R` | saemix reference (estimates in Comets 2026 paper) |
| `tests/reference/poisson_simulated/simulate.R` | 150 subjects; Poisson count; drug effect |
| `tests/reference/poisson_simulated/nonmem.ctl` | NONMEM F_FLAG=1, log-Poisson |

#### Must create before Phase 5 (CTMM)

| File | What it is |
|---|---|
| `tests/reference/ctmm_cav/get_data.R` | `data(cav, package="msm")` → ferx-format CSV |
| `tests/reference/ctmm_cav/msm.R` | msm reference fit (output already known — §7.6) |
| `tests/reference/ctmm_cav/mrgsolve_comparison.R` | mrgsolve ODE-based fit (output already known) |
| `tests/reference/ctmm_cav/expected.md` | Reference rates + OFV = 3968.798 |

#### Notes on NONMEM output files

The `.lst` files require a NONMEM 7.4+ license to generate, but **NONMEM availability does
not block development.** Workflow:
1. Commit `.ctl` and dataset now (reviewable without NONMEM)
2. Develop and validate against the free references (nlmixr2 / saemix / msm) **and** ferx's
   own SSE check (§8.8.8). This is sufficient to write the code, pass smoke tests, and open
   and merge exploratory PRs.
3. NONMEM results are **requested during development** and run externally as the license
   permits. When the `.lst` arrives, commit it and fold its numbers into `expected.md` as a
   verification step — applied as results arrive, before the endpoint is declared *fully
   validated*, not before coding starts.
4. The Tier 3 slow test reads reference values from `expected.md` and compares against
   whichever references are present (free tools always; NONMEM once added).

For phases where NONMEM is not the reference at all (CTMM, mCTMM), msm/mrgsolve output from
§7.6 is the verification gate and is already documented — no NONMEM step needed.

### 7.8 Reference comparison targets

| Model type | Primary comparison | Dataset |
|---|---|---|
| Parametric TTE | NONMEM LAPLACIAN=1 | Simulated (Holford tutorial) |
| Joint PK-TTE | NONMEM $DES + CHZ; nlmixr2 | Simulated; warfarin-like PK |
| RTTE | nlmixr2 SAEM; saemix | Holford 2013 RTTE dataset |
| Binary | saemix (toenail dataset) | de Backer 1998; in saemix R package |
| Ordinal | NONMEM F_FLAG=1 | Simulated 5-category dataset |
| Poisson / NB | NONMEM F_FLAG=1; `MASS::glm.nb` | Plan 2011 simulation |
| CTMM / mCTMM | R `msm` (CAV dataset) | Jackson 2011; in msm R package |
| DTMM | NONMEM (Bergstrand 2025 code) | Bergstrand supplementary |

---

## 8. Proposed Architecture

### 8.1 Data layer: polymorphic observation types

```rust
// src/types.rs

pub enum ObsRecord {
    Continuous { time: f64, value: f64, cmt: usize, mdv: bool },
    Event {
        time:           f64,
        event_type:     EventType,  // Exact, RightCensored, IntervalCensored { left, right }
        interval_start: f64,        // for RTTE; 0.0 for simple TTE
        entry_time:     f64,        // left truncation / delayed entry; 0.0 if none (§3.6)
        cmt:            usize,
    },
    DiscreteState { time: f64, state: usize, cmt: usize },  // CTMM/DTMM/ordinal/binary
    Count { time: f64, count: u32, cmt: usize },            // Poisson/NB
}

pub enum EventType {
    Exact,
    RightCensored,
    IntervalCensored { left: f64, right: f64 },  // both bounds explicit (consistent with §3.6)
}

// Subject: add parallel mixed observation Vec; keep existing Gaussian fields for
// backward-compatible path
pub struct Subject {
    // existing fields ...
    pub obs_records: Vec<ObsRecord>,  // populated from mixed-endpoint datasets
}
```

**ObsRecord → EndpointLikelihood routing.** The data reader picks the `ObsRecord` variant from
the endpoint **declared for that CMT** (the `[..._model]` block, §8.4) — never by guessing from
the DV value. One variant can serve several endpoints; the CMT's `EndpointLikelihood`
disambiguates:

| ObsRecord variant | Endpoint(s) declared for the CMT | DV column means |
|---|---|---|
| `Continuous` | `Gaussian` | observed value |
| `Event` | `TTE` (single or RTTE) | censoring code (0/1/2) + time |
| `DiscreteState` | `Binary`, `Ordinal`, `CTMM`, `DTMM` | category / state index |
| `Count` | `Poisson`, `NegBin` | non-negative integer count |

So `DiscreteState` with `state ∈ {0,1}` under a `Binary` endpoint is logistic regression; the
*same* variant under a `CTMM` endpoint is a Markov state observation. The reader must therefore
know each CMT's endpoint type **before** routing rows — a two-pass read: parse the `[..._model]`
blocks first, then dispatch each data row by its CMT.

### 8.2 Endpoint type in CompiledModel

```rust
pub enum EndpointLikelihood {
    Gaussian(EndpointError),                                          // existing
    TTE   { hazard: HazardSpec, rtte: Option<RtteClock> },  // None = single TTE; Some = RTTE (§3.3)
    CTMM  { q_fn: Box<QFn>, n_states: usize, time_homogeneous: bool },
    DTMM  { p_fn: Box<PFn>, n_states: usize },
    Binary  { link: LinkFn },
    Ordinal { cuts: Vec<f64>, link: LinkFn },
    Poisson { link: LinkFn },
    NegBin  { link: LinkFn, overdispersion: OverdispersionSpec },
    Custom  { ll_fn: Box<dyn Fn(f64, &PkParams, &[f64], &[f64]) -> f64 + Send + Sync> },
}

pub enum RtteClock { Forward, Reset }   // §3.3: total-time (Andersen–Gill) vs gap-time (renewal)

pub struct CompiledModel {
    // existing fields ...
    pub endpoints: HashMap<usize, EndpointLikelihood>,  // keyed by CMT value
}
```

### 8.3 Hazard specification

```rust
pub enum HazardSpec {
    Analytic {
        family:   HazardFamily,   // Exponential, Weibull, Gompertz, LogLogistic, LogNormal
        param_fn: Box<dyn Fn(&PkParams, &[f64], &[f64]) -> Vec<f64> + Send + Sync>,
    },
    OdeAccumulated {
        hazard_state_idx:  usize,  // ODE state holding ∫₀ᵗ h(u)du
        rate_eval_fn:      Box<dyn Fn(f64, &PkParams, &[f64], &[f64]) -> f64 + Send + Sync>,
    },
    Custom {
        rate_fn:  Box<dyn Fn(f64, &PkParams, &[f64], &[f64]) -> f64 + Send + Sync>,
        cumul_fn: Box<dyn Fn(f64, &PkParams, &[f64], &[f64]) -> f64 + Send + Sync>,
    },
}
```

### 8.4 DSL blocks

**TTE endpoint:**
```
[event_model]
cmt    = 2
type   = tte                  ; tte | rtte
clock  = forward              ; rtte only: forward (default) | reset   (§3.3)
family = weibull              ; exponential | weibull | gompertz | loglogistic | lognormal
shape  = THETA_SHAPE * exp(ETA_SHAPE)   ; α  (Weibull/loglogistic; omit for exponential)
scale  = THETA_SCALE                    ; λ  (baseline SCALE, a time: H=(t/λ)^α)
; --- covariate entry: choose PH or AFT (see §3.2) ---
loghr  = BETA_COV * COV       ; PH (default): hazard ×= exp(loghr); exp(β) is a hazard ratio
; AFT alternative — put the covariate on the scale and drop `loghr`:
;   scale = THETA_SCALE * exp(BETA_COV * COV)    ; exp(β) is a time ratio

; OR for ODE-linked hazard (overrides family/shape/scale):
chz_state = CHZ               ; name of ODE state accumulating H(t)
```
Note: do **not** put a covariate on Weibull `scale` and call it PH — that is AFT (§3.2). Use
`loghr` for PH. For Exponential/Gompertz the rate parameter *is* the hazard, so a covariate
there is genuinely PH.

**Ordinal model:**
```
[ordinal_model]
cmt    = 3
n_cats = 5
cuts   = ALPHA1, ALPHA2, ALPHA3, ALPHA4   ; auto-ordered: ALPHA1, then +exp(δ) (see §3.5)
logit  = BETA_DRUG * A1/V + ETA_LOGIT     ; linear predictor `lp`
; model is  P[Y<=k] = logistic(cut_k − lp)  → higher lp shifts mass to higher categories (§3.5)
```

**Count model:**
```
[count_model]
cmt    = 4
type   = poisson      ; or negbin
lambda = BASE * exp(BETA * A1/V) * exp(ETA)
; negbin only:
r      = R_OVERDISPERSION
```

**CTMM / mCTMM / DTMM:**
```
[markov_model]
cmt      = 5
type     = ctmm       ; ctmm, mctmm, dtmm
n_states = 3
; For ctmm — specify each non-zero rate:
q12 = LAMBDA12 * exp(ETA_12)
q21 = LAMBDA21
q13 = 0
q23 = LAMBDA23
q32 = LAMBDA32
; For mctmm — single parameter:
; tau = TAU_EQUIL    ; mean equilibration time = 1/q
; For dtmm — transition probabilities:
; p12 = logistic(ALPHA12 + BETA * Cc)
```

**Custom log-likelihood (escape hatch):**
```
[ll_model]
cmt = 6
ll  = -lambda + DV * log(lambda) - lgamma(DV + 1)
; Variables available: DV, TIME, and all model state outputs
```

### 8.5 Individual NLL dispatch

```rust
pub fn individual_nll(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
) -> f64 {
    let data_term: f64 = subject.obs_records_by_cmt()
        .map(|(cmt, records)| {
            match model.endpoints.get(&cmt) {
                Some(EndpointLikelihood::Gaussian(err))      => gaussian_data_term(...),
                Some(EndpointLikelihood::TTE { hazard, .. }) => tte_data_term(...),
                Some(EndpointLikelihood::Binary { link })    => binary_data_term(...),
                Some(EndpointLikelihood::Ordinal { cuts, .. }) => ordinal_data_term(...),
                Some(EndpointLikelihood::Poisson { link })   => poisson_data_term(...),
                Some(EndpointLikelihood::CTMM { q_fn, .. })  => ctmm_data_term(...),
                Some(EndpointLikelihood::DTMM { p_fn, .. })  => dtmm_data_term(...),
                Some(EndpointLikelihood::NegBin { .. })      => negbin_data_term(...),
                Some(EndpointLikelihood::Custom { ll_fn })   => custom_data_term(...),
                // Unregistered CMT is a model/data error — fail loudly, never silently
                // fall through to a default likelihood.
                None => panic!("no endpoint declared for CMT {cmt}"),
            }
        }).sum();
    let eta_prior = compute_eta_prior(eta, omega);
    data_term + 0.5 * (eta_prior + omega.log_det)
}
```

### 8.6 Outer Laplace for non-Gaussian (FD Hessian + log-det term)

The FOCEI function in `foce_subject_nll_interaction` dispatches:
- **Gaussian endpoints**: existing analytic Almquist formula (keep for performance)
- **Non-Gaussian endpoints**: FD Hessian with Shi step-size + `½ log|det(D_data + Ω⁻¹)|`

```rust
fn laplace_objective_nongaussian(
    model:   &CompiledModel,
    subject: &Subject,
    theta:   &[f64],
    eta_hat: &[f64],
    omega:   &OmegaMatrix,
) -> f64 {
    let data_term_at_mode = nll_data_term(model, subject, theta, eta_hat);
    let h_data = data_term_hessian_fd(|eta| nll_data_term(..., eta), eta_hat, &step_sizes);
    let h_total = h_data + omega.inv();                 // D_i^data + Ω⁻¹  (must be SPD at a min)
    // log|det H_total| via Cholesky = 2·Σ log L_ii. A Cholesky *failure* means H is not
    // positive-definite ⇒ η̂ is not a true minimum: signal a saddle (trigger §9.5 escape /
    // re-optimize), do NOT silently take |det| and continue.
    let log_det = match h_total.cholesky() {
        Some(c) => 2.0 * c.l().diagonal().iter().map(|d| d.ln()).sum::<f64>(),
        None    => return SADDLE_SENTINEL,   // non-PD: caller re-optimizes the inner problem
    };
    let eta_prior = eta_hat.dot(&omega.inv_times(eta_hat));
    data_term_at_mode + 0.5 * (eta_prior + omega.log_det + log_det)
}
```

### 8.7 CTMM matrix exponential module

New file `src/markov/mod.rs`:

```rust
/// 13th-order scaling-and-squaring Padé approximant (Higham 2005).
/// Stable for any Q with ‖Q‖ reasonable; rescale if ‖Q‖₁ > 1.
pub fn matrix_exp(a: &DMatrix<f64>) -> DMatrix<f64>;

/// Gradient of matrix_exp(Q) w.r.t. Q[j,k] using the Van Loan (1978) block trick:
///   ∂ expm(Q) / ∂ Q[j,k] = [expm([[Q, E_jk],[0,Q]])]_{1:S, S+1:2S}
/// Returns a slice of S×S gradient matrices, one per non-zero Q entry.
pub fn matrix_exp_grad(q: &DMatrix<f64>, entries: &[(usize,usize)]) -> Vec<DMatrix<f64>>;

/// CTMM individual data term: -Σ log P[s_{k+1} | s_k, Δt_k, Q(η,θ)]
pub fn ctmm_data_term(q: &DMatrix<f64>, records: &[(f64, usize)]) -> f64;

/// Time-inhomogeneous transition: solve dP/dt = Q(C(t))·P, P(0)=I using RK45.
/// Returns P(Δt) as S×S matrix.
pub fn ctmm_inhomogeneous_transition(
    q_at_c: impl Fn(f64) -> DMatrix<f64>,
    delta_t: f64,
    n_states: usize,
) -> DMatrix<f64>;

/// Forward algorithm for HMM (future phase): O(T×S²), log-sum-exp stable.
pub fn hmm_log_likelihood(
    transitions: &DMatrix<f64>,     // P(s'|s) row-stochastic
    emissions:   &[Vec<f64>],       // p(y_t | s) for each time
    init_dist:   &[f64],
) -> f64;
```

### 8.8 Simulation, Prediction & Diagnostics

**This is a first-class requirement, not an afterthought.** The current `simulate()` /
`predict()` path is hardcoded Gaussian. From `simulate_inner_with_draw` in `src/api.rs`:

```rust
let var = model.residual_variance_at(cmt, ipred, &sigma.values);
let dv_sim = ipred + var.sqrt() * eps;        // additive Gaussian — the entire sim model
```

and the result/prediction structs each carry a single continuous scalar:

```rust
pub struct SimulationResult { draw, sim, id, time, ipred: f64, dv_sim: f64 }
pub struct PredictionResult { id, time, pred: f64 }
```

Two structural assumptions break for non-Gaussian endpoints:

- **(A) Output ≠ prediction + noise.** Each endpoint has its own generative law (draw a
  category, draw a count, sample an event time, walk a chain). The `ipred + σ·ε` line must
  be replaced by an endpoint-dispatched sampler — mirroring the `individual_nll` dispatch
  (§8.5).
- **(B) For TTE / RTTE / CTMM the observation *times* are random outputs, not inputs.** The
  current loop iterates the input `obs_times` grid and adds noise at each. Event and path
  models generate their own timeline up to a horizon, then observe on a schedule. `time`
  flips from an input echo to a simulated output.

#### 8.8.1 Extended result types ✅ *SimulationResult merged — PR #190*

**As merged in PR #190** (`src/api.rs`, `src/types.rs`):

```rust
/// Unconditional variants (always compiled).  Feature-gated variants below.
#[derive(Debug, Clone)]
pub enum SimOutcome {
    Continuous { value: f64 },
    // Phase 4+: Category { state: usize }, Count { count: u32 }
    #[cfg(feature = "survival")]
    Event { time: f64, observed: bool },   // TTE/RTTE
}

impl SimOutcome {
    /// Returns the continuous DV value, or NAN for Event rows.
    pub fn continuous_value(&self) -> f64 { ... }
}

pub struct SimulationResult {
    pub draw:    usize,
    pub sim:     usize,
    pub id:      String,
    /// Scheduled obs time (Gaussian) or sampled event time (TTE).
    pub time:    f64,
    /// CMT that produced this row; 0 for all-Gaussian models.
    pub cmt:     usize,
    /// Individual prediction at η (Gaussian only; NAN for TTE).
    pub ipred:   f64,
    /// Simulated outcome — replaces the old `dv_sim: f64` field.
    pub outcome: SimOutcome,
}
```

**Breaking change:** `dv_sim: f64` field removed; callers must use
`row.outcome.continuous_value()`. Version bumped 0.1.5 → 0.1.6.

**Future variants** (`Category`, `Count`) will be added to `SimOutcome` unconditionally
(no feature flag) once Phase 4 lands. The `Event` variant will likewise be promoted
to default-on after Phase 1 validation.

The target `Prediction` enum and `SurvivalPredictionResult` remain as planned:

```rust
pub enum Prediction {
    Continuous { pred: f64 },
    Survival   { s: f64, cum_hazard: f64, hazard: f64 },  // TTE — Phase 1
    CatProbs   { probs: Vec<f64> },                       // categorical — Phase 4
    Rate       { lambda: f64 },                           // count — Phase 4
}
```

Prediction for the new types is a **curve or a probability vector**, never one scalar:
TTE → S(t)/H(t)/h(t) plus median/E[T]; categorical → P(Y=k|t); count → λ(t); CTMM →
occupancy π(t).

#### 8.8.2 Per-endpoint simulation

| Endpoint | Generative step | Fits fixed grid? | New infra |
|---|---|---|---|
| Gaussian | `f + σ·ε` (existing) | yes | — |
| Binary / ordinal | category probs → draw categorical | yes | sampler only |
| Poisson / NB | λ → draw Poisson/NegBin | yes | sampler only |
| TTE (analytic) | `u~U(0,1)`; `T = H⁻¹(−log u)` | **no** | inverse-CDF per family |
| TTE (ODE hazard) | integrate `dCHZ/dt=h`; stop when `CHZ = −log u` | **no** | **ODE event-location** (§8.8.3) |
| RTTE | repeat to `T_max`; reset hazard clock per event | **no** | horizon + CHZ reset (§8.8.4) |
| CTMM | Gillespie: `τ~Exp(−Q_ss)`; next ∝ off-diag; to `T_max` | **no** | **Gillespie engine** |
| DTMM | per fixed step: draw from `P` row | yes (stepwise) | row sampler |
| HMM | simulate hidden path, then emit | **no** | depends on CTMM engine |

Analytic inverse-CDF closed forms (no root-finding needed):
- Exponential: `T = −log(u) / λ`
- Weibull: `T = scale · (−log u)^(1/shape)`
- Gompertz: `T = (1/γ) · log(1 − (γ/α)·log u)`

#### 8.8.3 ODE event-location root-finder (shared infra — Phase 2)

For drug-driven hazard there is no closed-form inverse. Integrate the augmented ODE (which
already carries `dCHZ/dt = h(t)`) and **halt at the first t where CHZ(t) = −log(u)**. The
RK45 solver needs root detection: after each accepted step, test whether the monitored state
crossed the target; if so, locate the crossing by bisection / Hermite-interpolant root-find
within the step. This is the single largest new simulation piece and is reusable for any
threshold-crossing need.

```rust
// ode/solver.rs — new capability
pub fn integrate_until_threshold(
    deriv: impl Fn(f64, &[f64]) -> Vec<f64>,
    y0: &[f64], t0: f64, t_max: f64,
    monitor_state: usize, threshold: f64,
) -> Option<f64>;   // crossing time, or None if not reached by t_max → censor at t_max
```

#### 8.8.4 Simulation horizon & observation schedule

Event/path models need a per-subject horizon (Monolix's `rightCensoringTime`) and, for state
models, an observation schedule — you simulate the latent timeline, then observe it. New
simulation inputs:

```toml
[simulation]
horizon      = 48              # T_max: administrative censoring / stop time
obs_schedule = 0,6,12,24,48    # times at which CTMM/HMM state is recorded (optional)
```

This couples with the **RTTE clock type** (§3.3): under `clock = reset` (gap time) the hazard
accumulator must zero between events while PK compartments persist — the selective-reset
mechanism (§8.8.6), needed for *both* fitting and simulation. Under `clock = forward` (the
default, total time) there is no reset: the accumulator runs continuously and simulation just
keeps integrating to the horizon. Design the reset once in Phase 2; it is exercised only for
`clock = reset`.

#### 8.8.5 Diagnostics (sdtab) per endpoint

`{model}-sdtab.csv` currently emits IPRED/IWRES/CWRES — none of which is defined for a
non-Gaussian endpoint. Per-type standard diagnostics:

| Endpoint | IPRED analogue | Residual |
|---|---|---|
| TTE / RTTE | S(t), H(t) at EBE η | martingale `δ − H(t)`; deviance residual |
| Binary / ordinal | P(Y=k\|t) | standardized `(y − p)` |
| Count | λ(t) | Pearson `(y − λ)/√V` |
| CTMM | occupancy π(t) | state-prediction residual |

#### 8.8.6 Selective per-state ODE reset (for clock-reset RTTE only)

**Only `clock = reset` (gap-time) RTTE needs this** — see §3.3. There, the cumulative-hazard
accumulator must zero between events **without** disturbing PK compartments. `Subject::reset_times`
currently implements EVID=3 semantics — it zeros *all* compartments. Phase 2 adds a per-state
reset (reset a named ODE state, leave the rest), used by:
- the clock-reset RTTE NLL: `Σ_k log h(Δ_k) − Σ_k H(Δ_k)` over inter-event gaps, and
- the clock-reset RTTE simulator: continue the PK ODE across events while restarting CHZ each gap.

Clock-forward (default) RTTE needs **no** reset — the accumulator runs continuously and the NLL
is `Σ_k log h(t_k) − H(T)`. Alternative implementation for clock-reset (no in-place reset):
integrate h over each inter-event gap as an independent sub-integration and sum — equivalent,
and simpler if the selective reset proves fragile. Decide in Phase 2.

#### 8.8.7 VPC

Every VPC type requires the simulation engine above to generate replicate datasets that
respect the original censoring/observation structure:
- TTE: Kaplan–Meier VPC (simulated KM curves + CI band vs. observed KM)
- RTTE: mean cumulative function (Nelson–Aalen) per event number
- Categorical: category-proportion bands over time
- Count: PMF / mean-count bands

These are downstream tooling (likely ferx-r side), but the *simulation primitive* they depend
on is ferx-core's responsibility.

#### 8.8.8 Simulation-estimation (SSE) — license-free validation

Once an endpoint can both fit and simulate, the strongest self-contained test is available:
**simulate from known (θ, Ω) with ferx → fit with ferx → confirm recovery within MC error.**
This needs no NONMEM/nlmixr2/msm license and should be a Tier 3 test for every endpoint,
complementing the external-reference comparisons of §11. It also guards the new generative
code paths against silent bugs that the fitting tests cannot see.

---

## 9. Methods Improvements (What ferx Can Implement Beyond NONMEM/Monolix)

### 9.1 f-SAEM: Laplace-proposal MH — user option with guidance

**What it is:** An alternative MH proposal for the SAEM E-step. Instead of a random-walk
proposal, an independent Gaussian centered at the current EBE η̂ with covariance H⁻¹
(from the inner BFGS Hessian) is used.

**Why it is an option, not a replacement:**

f-SAEM is faster when the individual posterior is approximately Gaussian at η̂ — which
is true for well-determined subjects (many observations, strong PK-TTE link). In that
regime: no tuning required, zero autocorrelation between proposed samples, 3–5× fewer
samples needed per iteration (vs. 10–20 for random-walk with 5 ETAs).

However, f-SAEM has a genuine failure mode: when the individual posterior is far from
Gaussian — exactly the difficult cases where SAEM is most needed (RTTE with <1 event/
subject, sparse binary data, early SAEM iterations before θ has converged) — the Laplace
proposal is poorly calibrated and acceptance rates can collapse toward zero, stalling the
chain. Random-walk MH mixes slowly in those cases but does not get stuck.

Neither proposal is universally superior; the optimal choice is subject- and
model-dependent.

**Proposed interface:**

```toml
[fit_options]
saem_proposal = auto    # default; options: laplace | random_walk | auto
```

| Value | Behaviour | Recommended when |
|---|---|---|
| `laplace` | Always use Laplace (f-SAEM) independent proposal | Well-identified subjects; joint PK-TTE with strong PK anchor |
| `random_walk` | Always use adaptive random-walk MH (current behaviour) | RTTE with sparse events; pure TTE; early-phase estimation |
| `auto` | Monitor per-subject acceptance rate over burn-in; switch to random-walk for any subject whose acceptance drops below 0.05 | **Default — best general choice** |

`auto` is the new default: it behaves like Monolix (which uses f-SAEM internally with
built-in fallback, without exposing the switch to users).

**Implementation sketch:**

```rust
// In estimation/saem.rs, E-step for subject i:

let (eta_hat, h_bfgs) = inner_bfgs_with_hessian(...);

let use_laplace = match options.saem_proposal {
    SaemProposal::Laplace    => true,
    SaemProposal::RandomWalk => false,
    SaemProposal::Auto       => subject_acceptance_rate[i] > 0.05,
};

if use_laplace {
    let proposal_cov = h_bfgs.try_inverse()
        .unwrap_or_else(|| DMatrix::identity(n_eta, n_eta) * 0.1);
    let chol = proposal_cov.cholesky()
        .unwrap_or_else(|| /* fallback to random-walk */ ...);
    // independent Gaussian proposal
    let eta_proposed = &eta_hat + chol.l() * standard_normal_vector(n_eta, rng);
} else {
    // existing adaptive random-walk MH
}
// update subject_acceptance_rate[i] via exponential moving average
```

The Hessian comes from the inner BFGS accumulation at convergence — zero extra cost.
The per-subject acceptance rate tracker adds O(N) memory, negligible.

### 9.2 Adaptive Gauss-Hermite Quadrature (AGQ) — optional feature

For models with q ≤ 3 random effects and non-Gaussian likelihoods, AGQ reduces Laplace
approximation bias, especially for binary/categorical endpoints where the individual
posterior is far from Gaussian.

**Algorithm:**
1. Find mode η̂ and Hessian H at mode (already available from inner BFGS)
2. Map the integral using the Cholesky: `∫ f(η) dη ≈ det(L) · ∫ f(L·u + η̂) du`
3. Apply Gauss-Hermite quadrature in the rotated space with k nodes per dimension:
   `≈ det(L) · Σ_{u∈grid} f(L·u + η̂) · w(u)`

For k=3, q=2: 9 evaluations per subject. For k=3, q=3: 27 evaluations.
For k=3, q=5: 243 evaluations — borderline practical.

**Implementation:** Gate behind `--features agq`; expose via `[fit_options] n_agq = 3`.

### 9.3 FD step-size tuning (Shi 2021 algorithm)

For the FD Hessian of non-Gaussian data terms, the step size ε should be adaptive:

```rust
fn shi_step_size(gradient_norms: &[f64]) -> f64 {
    // Harmonic mean of per-observation gradient norms at η̂
    let n = gradient_norms.len() as f64;
    let harmonic = n / gradient_norms.iter().map(|g| 1.0 / g.max(1e-10)).sum::<f64>();
    harmonic.powf(1.0 / 3.0) * f64::EPSILON.powf(1.0 / 3.0)
}
```

This produces step sizes that are ~10× better conditioned than fixed ε=1e-5 for
typical pharmacometric non-Gaussian likelihoods (nlmixr2 validation).

### 9.4 BHHH (gauss_newton.rs) for non-Gaussian — works as-is

**Key insight from research:** The BHHH (Berndt-Hall-Hall-Hausman) approximation
`H ≈ Σᵢ gᵢgᵢᵀ` is valid for any well-specified probability model because the
Fisher information identity `E[∇l·∇lᵀ] = -E[∇²l]` holds under correct model
specification. The BHHH outer loop in `estimation/gauss_newton.rs` does NOT need
changes — only the per-subject `nll` function it calls.

Once `individual_nll` dispatches to non-Gaussian data terms, GN-Hybrid will
automatically work for TTE, ordinal, Poisson, and CTMM models.

Note: Pure GN uses the Gaussian residual structure to compute J'R⁻¹J analytically.
This does not extend to non-Gaussian. GN-Hybrid (BHHH outer + FOCEI polish) is the
right mode for non-Gaussian.

### 9.5 Saddle-point detection for inner BFGS

For non-Gaussian individual likelihoods (especially TTE at extreme censoring rates or
ordinal with sparse categories), the individual log-posterior can be multimodal or have
saddle points that trap the BFGS optimizer.

**Detection:** After BFGS convergence, compute eigenvalues of the final BFGS Hessian
approximation. If the minimum eigenvalue `λ_min < -ε`, a saddle point has been found.

**Recovery:**
```rust
fn saddle_point_escape(eta: &mut Vec<f64>, hess: &DMatrix<f64>, rng: &mut impl Rng) {
    let decomp = hess.symmetric_eigen();
    let min_idx = decomp.eigenvalues.iter()
        .enumerate().min_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    if decomp.eigenvalues[min_idx] < -1e-6 {
        let v = decomp.eigenvectors.column(min_idx);
        let step = 0.1 * rng.sample::<f64, _>(Uniform(-1.0, 1.0)).signum();
        for j in 0..eta.len() { eta[j] += step * v[j]; }
        // Restart BFGS from perturbed point
    }
}
```

### 9.6 Corrected Integrated Laplace (CILA, 2025)

For the outer objective, the standard Laplace approximation has O(1/n) error per
subject. CILA (arXiv:2605.20345) corrects this via importance sampling:

```
π̃(θ|y) = (1/n) Σᵢ p(y|zᵢ,θ) p(zᵢ|θ) / N(zᵢ; η̂, H⁻¹)
```

where `z₁...zₙ ~ N(η̂, H⁻¹)` are drawn from the Laplace approximation. This is an
**unbiased** estimator of the marginal likelihood. The QMC variant achieves O(n⁻²)
variance using quasi-random sequences.

**Practical recommendation:** For ferx, CILA is most valuable for small datasets
(N < 30 subjects) with non-Gaussian likelihoods where Laplace bias is clinically
significant. Expose as `[fit_options] cila_samples = 50`.

### 9.7 Van Loan block trick for CTMM gradients

For computing `∂ expm(Q) / ∂ Q[j,k]` without finite differences:

```rust
fn matrix_exp_param_grad(q: &DMatrix<f64>, j: usize, k: usize) -> DMatrix<f64> {
    let s = q.nrows();
    let mut c = DMatrix::zeros(2*s, 2*s);
    c.view_mut((0,0), (s,s)).copy_from(q);
    c.view_mut((s,s), (s,s)).copy_from(q);
    c[(j, s + k)] = 1.0;   // E_jk in the upper-right block
    let ec = matrix_exp(&c);
    ec.view((0,s), (s,s)).into_owned()  // upper-right block = the gradient
}
```

One 2S×2S matrix exponential per parameter gives the exact gradient. For S=3 and
5 free parameters: 5 matrix exponentials of a 6×6 matrix — negligible cost.

---

## 10. Adjacent Field Insights

### 10.1 Frailty models as the pharmacometric standard

In survival analysis, the **shared frailty model** is:
```
h(t | ωᵢ) = ωᵢ · h₀(t) · exp(β'Xᵢ)
```

Two frailty distributions:
- **Gamma frailty** (`ωᵢ ~ Gamma(1/θ, 1/θ)`): closed-form marginal likelihood via
  Laplace transform; analytically tractable but misspecified if true distribution ≠ Gamma.
- **Log-normal frailty** (`ωᵢ = exp(ηᵢ)`, `ηᵢ ~ N(0, ω²)`): standard in pharmacometrics;
  same as the NLME random-effect structure; requires numerical integration.

In ferx, log-normal frailty is automatic — the ETA on the hazard parameter IS the frailty.
The gamma frailty marginal would be a speed optimization: compute the marginal
analytically, bypassing the inner BFGS. Relevant only if fitting TTE models without PK
(no other reason to have a random effect).

**Identifiability caveat (document prominently).** In a *standalone single-event* TTE model,
each subject contributes at most one event, which carries almost no information about
*between-subject* variance. The frailty variance ω is therefore weakly (often non-)
identified — the inner EBEs collapse toward 0 and ω drifts to a boundary. This is a
statistical limitation of the data, not a ferx defect. Practical guidance to surface in the
docs and (where detectable) as a `FitResult` warning:

- TTE random effects are well-identified in **joint PK-TTE** (the PK observations inform η)
  and in **RTTE** (multiple events per subject) — these are the intended homes for frailty.
- For standalone single-event TTE, prefer a **fixed-effects** hazard (no ETA) unless there is
  a strong external reason and many events; if ω is estimated, report it with a caveat and
  check the ω profile/SE for non-identifiability.

### 10.2 Competing risks

Two formulations:

| Approach | What it models | Hazard ratio interpretation | Complexity |
|---|---|---|---|
| Cause-specific hazard | Risk among currently event-free subjects | Straightforward | Low — each event type is a separate TTE model |
| Fine-Gray subdistribution | Cumulative incidence function (CIF) | Risk in full population | High — IPCW weighting; numerically unstable for sparse data |

**Recommendation for ferx:** Implement cause-specific hazard only. Each event type k
has its own `HazardSpec`; subjects experiencing another event type are censored for
event type k. The data format uses CMT to distinguish event types:

```csv
ID,TIME,DV,CMT,...
1,24,1,2,...   ; event type 1 at t=24 (CMT=2)
1,24,0,3,...   ; censored for event type 2 at t=24 (CMT=3)
```

A `[event_model]` declaration per CMT specifies the hazard for each event type.

### 10.3 Cox partial likelihood (semi-parametric)

The Cox proportional hazard model makes no assumption about the baseline hazard h₀(t),
eliminating a potential source of model misspecification. The partial likelihood is:

```
PL(β) = Π_k exp(β'Xᵢₖ) / Σ_{j in risk set at t_k} exp(β'Xⱼ)
```

**Why semi-parametric Cox does not fit ferx's architecture.** The whole estimation engine
rests on a likelihood that decomposes into *independent per-subject terms*,
`Σᵢ individual_nll(ηᵢ)`, with the inner loop solving each subject in isolation. The partial
likelihood does **not** decompose this way: the denominator is the risk set `R(t_k)` — every
subject still at risk at event time `t_k` — so subject *i*'s contribution depends on all other
subjects. That coupling breaks the independent-subject structure the inner/outer loops require.
Additional friction: PK covariates are time-varying (from the ODE), so the risk set must be
re-evaluated (and every at-risk subject's ODE integrated) at each event time; mixed-effects Cox
(frailty) needs penalized partial likelihood / h-likelihood, not FOCEI/SAEM; and with no
baseline hazard there is no `S(t)` for prediction/simulation without a Breslow post-step.

**Feasible alternative — flexible parametric baseline (recommended over Cox).** The usual
reason to reach for Cox ("don't assume a hazard shape") is met *inside* the per-subject
likelihood by a flexible parametric baseline, which stays fully compatible with the current
setup because each subject's `H(t)` remains its own integral:

- **Piecewise-constant hazard** (exponential within time intervals): trivial in `HazardSpec`
  (`family = piecewise` with interval breakpoints + per-interval log-hazard θ's). This is the
  standard pharmacometric stand-in for Cox and converges to it as intervals shrink.
- **Spline / Royston–Parmar** baseline on `log H(t)` (flexsurv's `flexsurvspline`): smooth,
  few parameters, approximates an arbitrary baseline. A natural Phase 8 add to `HazardSpec`.

Both are PH (or AFT) parametric models — estimable today with Laplace/SAEM, no new machinery.

**Recommendation:** Defer *semi-parametric* Cox (partial likelihood) — it is architecturally
incompatible and rarely needed. Cover the same modeling intent with the piecewise-constant /
spline baseline above. Flag genuine partial-likelihood Cox as a potential Phase 8 only if a
concrete use case demands it.

### 10.4 Neural survival models — ideas for ferx

DeepSurv and survival random forests are ML approaches but contain transferable ideas:
- **Monotone hazard parameterization**: parameterize log H(t) directly as a monotone
  neural network (Weibull is a special case). Relevant only if offering non-parametric
  baseline hazard.
- **Permutation importance**: useful as a post-hoc covariate ranking tool for TTE models.

Not directly implementable, but the ODE-as-cumulative-hazard (`HazardSpec::OdeAccumulated`,
§8.3; Phase 2) is the pharmacometric equivalent of these approaches — it learns a flexible
hazard shape through the PK ODE.

### 10.5 IRT (Item Response Theory) as generalization of proportional odds

Standard proportional odds (cumulative logit) assumes equal item discrimination `a=1`.
The Graded Response Model (GRM) relaxes this:

```
P(Y ≥ s) = logistic(a · (θᵢ − b_s))
```

where `a` is item discrimination, `b_s` is category difficulty, `θᵢ` is the subject's
latent trait (driven by drug PK/PD). IRT transforms ordinal scale to interval scale and
provides implicit item weighting.

**Relevance:** Composite symptom scores (e.g., HAM-D, UPDRS, PANSS) are often modeled
via proportional odds in pharmacometrics. IRT offers a theoretically superior alternative
at the cost of estimating per-item discrimination parameters (requires item-level data).

**Implementation path:** IRT is a natural Phase 5+ extension of the ordinal model
(Phase 4) — same DSL structure with an added item-discrimination parameter.

### 10.6 INLA for joint longitudinal-survival

INLA (Integrated Nested Laplace Approximation) is a deterministic Bayesian method for
latent Gaussian models that is 100–1000× faster than MCMC. For the pharmacometric use
case:
- The latent field (ETAs) is Gaussian — INLA's requirement satisfied
- INLA computes the posterior on a grid over hyperparameters (θ, Ω, σ)
- Each grid evaluation requires one Laplace approximation — similar cost to FOCE

The `INLAjoint` R package supports joint longitudinal-survival models. INLA is not
applicable to ferx's outer BOBYQA optimization (INLA is Bayesian; ferx is frequentist
MLE) but it provides a fast Bayesian validation benchmark for TTE/Markov models.

### 10.7 Scalable CTMM for large state spaces (Krylov vs Padé)

| Method | Best for | Memory | Runtime |
|---|---|---|---|
| Padé (full) | S ≤ 10 | O(S²) | O(S³) |
| Krylov / Arnoldi | S large, sparse Q | O(m·S) | O(S·m²) |
| Power series | S ≤ 20, short Δt | O(S²) | O(S³·n_terms) |
| SGD + block-Padé (SCTMM) | N > 1000, S ≤ 20 | O(S²) | Amortized O(S³·|B|) |

For ferx initial implementation: Padé is correct choice for S ≤ 10 states. Krylov
(via Expokit algorithm, Sidje 1998) becomes relevant for S ≥ 10. Flag as `#[cfg(feature = "markov")]`.

### 10.8 Full Bayesian via Stan/Torsten (comparison target)

Stan + Torsten (PK-specific Stan library) provides:
- Analytical PK solvers compatible with HMC
- Full posterior via NUTS — no approximation error
- Reverse-mode AD through ODE solutions for cumulative hazard H

Stan is 10–100× slower than FOCE for large populations but provides exact posterior
estimates. For ferx validation of non-Gaussian models, Stan serves as a gold-standard
reference when NONMEM results are uncertain.

For a Weibull TTE model in Stan:
```stan
model {
    for (i in 1:N) {
        if (event[i] == 1)
            target += weibull_lpdf(T[i] | alpha, lambda * exp(-beta * Cc[i]));
        else
            target += weibull_lccdf(T[i] | alpha, lambda * exp(-beta * Cc[i]));
        target += normal_lpdf(eta[i] | 0, omega);
    }
}
```

### 10.9 VAE-NLME — amortized inference

The VAE-NLME framework (arXiv:2601.17400) trains an encoder network that maps each
subject's observation vector to posterior parameters (mean, covariance) in a single
forward pass. After training, new subject EBEs are computed instantly — relevant for
real-time TDM (therapeutic drug monitoring) or large-scale simulation.

Currently restricted to Gaussian likelihoods; non-Gaussian extension requires normalizing
flows or discrete VAE on the observation model. **Not a near-term ferx feature** but worth
tracking — this could replace the inner BFGS loop entirely for online/streaming applications.

---

## 11. Development Process — Required for Every Phase

Each phase follows a mandatory validation-driven loop. **No phase is complete until
every step in the loop passes.** The loop is not optional or post-hoc — it is the
definition of "done" for each feature.

### 11.1 The loop (apply to every phase, every sub-feature)

```
1. Reference first — and NONMEM is NON-BLOCKING
   ├── Write simulate.R and generate the dataset (fixed seed, committed)
   ├── Write ALL reference scripts up front: free tools (nlmixr2 / saemix / msm) AND the
   │   NONMEM .ctl — commit the .ctl now even though its .lst comes later
   ├── Run the FREE references immediately → commit their output; these are the interim
   │   primary reference and are sufficient to develop, smoke-test, and open PRs
   ├── Add ferx's own SSE check (§8.8.8) as a second license-free anchor
   └── Fill expected.md from the free references now; NONMEM columns are added when its
       results arrive (see step 1b) — never wait on NONMEM to start

1b. NONMEM verification — requested DURING development, applied when available
   ├── NONMEM results are REQUESTED during development and run externally as the license
   │   permits; they are a verification gate, not a precondition for starting or merging
   ├── When the .lst arrives: commit it, add OFV/theta/omega/SE to expected.md, and confirm
   │   ferx matches within the §11 tolerances
   └── An endpoint is "fully validated" once the NONMEM comparison is in; it is "developed
       and mergeable" before that, on the strength of the free references + SSE

2. Implement in ferx
   ├── Write the Rust code
   ├── cargo check (verify compilation, no warnings)
   └── Write Tier 2 smoke test: fit() returns Ok in ≤ 5 outer iterations

3. Compare against reference
   ├── Run ferx on the identical dataset used by the reference
   ├── OFV: within 0.5 units of NONMEM (or within 1.0 for msm/nlmixr2)
   ├── Parameter estimates (theta, omega): within 10% of reference
   └── Standard errors: within 20% of reference

4. Evaluate discrepancies
   ├── If OFV matches but estimates differ: check parameterization transforms
   ├── If OFV is off by a constant: check the ½ log|2πΩ| or log|det H| term
   ├── If OFV diverges with N: check per-observation likelihood sign / scaling
   ├── If SEs are wrong: check covariance step (sandwich vs. inverse Hessian)
   └── If SAEM doesn't converge: check sigma update (must be skipped for non-Gaussian)

5. Fix and recompare
   └── Repeat steps 2–4 until all tolerances are met

6. Write Tier 3 convergence test
   ├── Only after comparison passes
   ├── Gates on #[cfg_attr(not(feature = "slow-tests"), ignore = "slow")]
   └── Reads expected values from expected.md; fails if ferx drifts

7. Add comparison table to docs
   ├── Required by CLAUDE.md policy for any numerical result
   ├── Table: dataset description | ferx OFV | reference OFV | key estimates side-by-side
   └── Note any known acceptable discrepancies (e.g., constant OFV offset due to
       normalizing constants that NONMEM omits)

8. PR merge gate — all of the following must be present:
   ├── expected.md committed with reference values (free references + SSE suffice for merge;
   │   the NONMEM column is added later per step 1b and does NOT gate the PR)
   ├── Tier 2 smoke test passing in CI
   ├── Comparison table in docs
   └── Tier 3 convergence test written (run nightly, not blocking PR)

9. Plan ferx-r integration (triggered when step 6 passes)
   ├── Open a scoped plan document in ferx-r describing the R-side API:
   │     - How the user specifies this model type in R (new arguments,
   │       new DSL helpers, or new fit_options keys)
   │     - Any new or changed R functions / S3 methods needed
   │     - Data format: how censoring / state / count columns are passed
   │       from an R data frame into the ferx binary
   │     - Vignette or pkgdown example page to add
   ├── Identify changes to the ferx-r Rust glue layer:
   │     - New public types in ferx-core that must be re-exported
   │     - New fields on FitResult, FitOptions, CompiledModel exposed to R
   │     - Breaking changes to any existing public API
   ├── Write the ferx-r PR alongside or immediately after the ferx-core PR —
   │   not deferred indefinitely; the two repos must stay in sync
   └── ferx-r PR requires its own end-to-end comparison:
         run the R reference script (nlmixr2 / saemix / msm) and the ferx-r
         wrapper on the same dataset; confirm output matches the ferx-core
         standalone binary result from step 3
```

### 11.2 Common failure modes to check first

| Symptom | Likely cause |
|---|---|
| OFV off by a fixed constant (~const·N) | Missing or double-counted `½ log(2π)` normalizing term |
| OFV off by `½ log\|Ω\|` | FOCE path used instead of Laplace; log-det term missing |
| OFV sign-flipped | `nll` returned instead of `ll`; check sign conventions |
| Estimates match NONMEM but SEs are ~2× off | covariance step using wrong Hessian (FD vs. analytic) |
| TTE OFV matches, but variance (omega) is biased low | FOCE used for non-Gaussian (missing `½ log\|det H_data\|` term) |
| RTTE omega severely biased (−90%) | Laplace used for sparse RTTE — switch to SAEM |
| SAEM doesn't improve after 50 iterations | sigma update not disabled for non-Gaussian endpoint |
| CTMM OFV drifts from msm | Matrix exponential instability for large Q — check rescaling |
| Ordinal SEs much larger than reference | Cut-points not identifiable — constrain one intercept |

### 11.3 Acceptable discrepancies (document, do not fix)

Some differences from NONMEM are expected and should be documented rather than
treated as bugs:

- **Normalizing constants**: NONMEM omits certain constants (e.g., `N·log(2π)`)
  from the OFV for speed. ferx should be consistent internally; document the offset.
- **Random-effects scaling**: NONMEM sometimes reports `2·NLL` while ferx reports
  `NLL`; confirm the factor-of-2 convention per model type.
- **SAEM stochasticity**: SAEM results have Monte Carlo variance (~0.1–0.5 OFV units);
  the Tier 3 test should allow ±1.0 OFV units for SAEM comparisons, ±0.5 for FOCEI.
- **msm vs. ferx CTMM**: msm uses a different parameterization of the rate matrix
  (rows vs. columns); confirm transition direction convention and document if it differs.

---

## 12. Implementation Phases

### Phase 1 — Parametric TTE, standalone, Laplace

**Scope:** Exponential, Weibull, and Gompertz; fixed and random hazard parameters; FOCEI
Laplace; right-censored, interval-censored, and **left-truncated (delayed entry)**; no PK.

**Status: complete.** All three Phase 1 ferx-core PRs merged: #190 (infrastructure scaffold),
#192 (wiring, 2026-06-06), and #206 (follow-up, 2026-06-09 — IOV+TTE, SAEM+TTE, optional
blocks, covariate tracking, median/mean survival, BIC + warning fixes). ferx-r TTE routing
also complete: PR #134 (initial routing) and PR #142 (final consolidation through
`read_population_for`) both merged 2026-06-09. Only Tier 3 slow-tests, NONMEM comparison,
and the `predict_survival` R wrapper remain.

#### Done — PR #190 (infrastructure scaffold)

- ✅ `survival = []` feature flag in `Cargo.toml`; version 0.1.5 → 0.1.6
- ✅ **`SimulationResult` redesigned** (breaking): `dv_sim: f64` removed; `outcome: SimOutcome`
  + `cmt: usize` added. `SimOutcome::Continuous` unconditional; `SimOutcome::Event` gated on
  `survival`. `continuous_value()` returns NAN for Event rows with a debug_assert guard.
- ✅ New types in `src/types.rs`: `EventType`, `ObsRecord`, `HazardFamily`, `HazardParamFn`,
  `HazardSpec::Analytic`, `EndpointLikelihood::Tte` — all `#[cfg(feature = "survival")]`
- ✅ `Subject.obs_records: Vec<ObsRecord>` and `CompiledModel.endpoints: HashMap<usize,
  EndpointLikelihood>` — cfg-gated, empty in all existing builds (no overhead)
- ✅ `src/survival/parametric.rs` — `hazard_and_cum_hazard`, `cum_hazard`, `sample_event_time`,
  `sample_conditional_event_time` for Exponential, Weibull, Gompertz; full Tier 1 test suite
- ✅ `src/survival/mod.rs` — `tte_data_term` (all EventType variants + left truncation);
  `data_term_hessian_fd` (4-point central stencil); `shi_step_sizes` (Shi 2021 §3.4);
  `simulate_tte` (draws event times; called from `api::simulate_inner_with_draw`)
- ✅ Reference files: `tests/reference/tte_exponential/`, `tte_weibull/`, `tte_gompertz/`
  (simulate.R, nlmixr2.R, nonmem.ctl, expected.md for each)

#### Done — PR #192 (wiring, merged 2026-06-06)

**Parser (`src/parser/model_parser.rs`):**
- ✅ `[event_model]` block parsing — `param_fn` closure for Exponential, Weibull, Gompertz;
  keys: `cmt`, `family`, `scale`/`rate`, `shape`, `alpha`, `gamma`, `loghr` (optional PH term)
- ✅ Named blocks (`[event_model NAME]`) for multiple TTE endpoints; duplicate-CMT guard
- ✅ Incompatible key validation (e.g. `shape` in `exponential` → parse error)
- ✅ `n_eta=0` fix: `build_omega_matrix` returns 0×0 Omega when no etas declared

**Datareader (`src/io/datareader.rs`):**
- ✅ `TENTRY` column auto-detected; DV=0/1/2 routed to `subject.obs_records` via
  deferred-flush pattern; end-of-subject flush for remaining pending left bounds
- ✅ Non-integer DV on TTE CMT → hard parse error (previously silently truncated)
- ✅ `TENTRY > TIME` → parse warning + row skip (previously silent negative cumulative hazard)
- ✅ Dead `#[cfg(not(feature="survival"))]` fallback branch removed (was unreachable)

**Likelihood (`src/stats/likelihood.rs`):**
- ✅ `individual_nll_into_with_schedule`: TTE data term added (2× scaling to match Gaussian
  halving convention); iterates `model.endpoints` directly (no `HashSet` scan)
- ✅ `foce_subject_nll_interaction_with_tte`: FD Hessian + `½ log|det H_total|` for TTE CMTs;
  seeds TTE NLL + Hessian into combined Laplace correction; iterates `model.endpoints`

**API (`src/api.rs`):**
- ✅ `read_population_for` promoted to `pub` — single entry point handling covariates,
  `[data_selection]` filters, and TTE routing; the function external consumers should call
- ✅ `predict_survival` + `SurvivalPredictionResult`: S(t), H(t), h(t) on a time grid per
  subject × TTE CMT, plus `median_survival` and `mean_survival` fields

**Tests (`tests/tte_smoke.rs`, Tier 2):**
- ✅ `tte_exponential_model_parses`, `tte_weibull_model_parses`, `tte_gompertz_model_parses`
  — all three parser branches covered
- ✅ `tte_fixed_effects_model_parses` (n_eta=0 path), `tte_fit_exponential_3iter`,
  `tte_fit_fixed_effects_n_eta_0`, `tte_loghr_nonzero_changes_ofv` (OFV shift > 1.0),
  `tte_duplicate_cmt_parse_error`, `tte_incompatible_key_*` (two error cases)
- ✅ `tte_only_model_parses_without_pk_blocks`, `tte_only_fit_completes_without_pk_blocks`,
  `event_model_covariate_names_tracked`, `predict_survival_has_median_and_mean`

**Docs:**
- ✅ `docs/estimation/tte.qmd` — overview, syntax, DV coding, TENTRY, hazard families
  table (with `exp(loghr)` multiplier), loghr examples, estimation notes, placeholder
  NONMEM/nlmixr2 comparison table
- ✅ `docs/model-file/event-model.qmd` — `[event_model]` key reference including
  `loghr` and `rate`; expression namespace Note callout
- ✅ `docs/_quarto.yml` sidebar updated

**Examples + data:**
- ✅ `examples/tte_exponential.ferx` (using correct theta/eta expressions)
- ✅ `examples/tte_weibull.ferx`, `examples/tte_gompertz.ferx` (clean TTE-only syntax)
- ✅ `data/tte_exponential.csv` (30-subject simulated dataset)

#### Done — PR #206 (follow-up, merged 2026-06-09)

**Parser / DSL:**
- ✅ `[individual_parameters]`, `[structural_model]`, `[error_model]` blocks optional when
  `[event_model]` is the sole endpoint — TTE-only `.ferx` files parse and fit without them
- ✅ Covariate names referenced inside `[event_model]` expressions propagated into
  `CompiledModel.referenced_covariates` — `event_model_covariate_names_tracked` test
- ✅ False "parameter not referenced in [individual_parameters]" warnings suppressed for
  TTE-only models (`check_unused_parameters` early-returns when `has_event_model && indiv_stmts.is_empty()`)

**Estimation / API:**
- ✅ BIC finite for TTE-only models — `n_for_bic` uses TTE `obs_records` count when
  `n_obs == 0`; prevents `ln(0) = -inf`
- ✅ SAEM M-step TTE term — `obs_nll_subject_into` now adds `tte_data_term` for each
  `EndpointLikelihood::Tte` endpoint; previously SAEM theta estimates for TTE were wrong
- ✅ SAEM TTE weighting corrected — 2× scaling factor consistent with FOCEI halving convention
- ✅ IOV + TTE joint support — `individual_nll_into_with_schedule` handles kappa draws
  alongside TTE; `foce_subject_nll_interaction_with_tte` updated for IOV subjects;
  Tier-2 smoke test `tte_iov_subjects_fit_3iter` added
- ✅ Gompertz γ=0 edge cases — simulation and `mean_survival` guarded against division-by-zero
- ✅ `median_survival` / `mean_survival` on `SurvivalPredictionResult` — analytic closed-form
  for Exponential and Weibull median; numerical midpoint for Gompertz mean
- ✅ `predict_survival` + `SurvivalPredictionResult` re-exported at crate root

**Examples + data:**
- ✅ `data/tte_weibull.csv` (30-subject simulated Weibull dataset)
- ✅ `data/tte_gompertz.csv` (50-subject, BSV on gamma, 80 h censoring, 42/50 events)
- ✅ `examples/tte_gompertz.ferx` redesigned — BSV on gamma (not alpha) for clean recovery;
  TVALPHA=0.002038, TVGAMMA=0.051491 vs truth 0.002/0.05; collinearity note in header

**Code quality:**
- ✅ `foce_subject_nll_interaction_with_tte` refactored — `gaussian_foce_accum` helper
  eliminates the duplicated inner loop from both FOCE functions

**ferx-r:**
- ✅ `r.dv_sim` → `r.outcome.continuous_value()` migration — ferx-r PR #132 (merged 2026-06-08)
- ✅ TTE datareader routing — ferx-r PR #134 (merged 2026-06-08) + PR #142 (merged 2026-06-09);
  all callsites consolidated through `read_population_for`; TTE obs_records now land correctly
  for `ferx_fit`, `ferx_simulate`, and `ferx_predict`

#### Remaining — deferred from Phase 1

**Estimation:**
- ✅ Tier 3 convergence + SSE tests (`tests/tte_convergence.rs`, gated `survival,slow-tests`) — **all
  three families done** (branch `test/tte-phase1-validation`, 8 tests): Exponential SSE (N=2000 →
  λ −1%, ω² −7%) + mixed + fixed-vs-`survreg` (**exact**: 0.074506, OFV 589.888); Weibull SSE +
  mixed + fixed-vs-`survreg` (**exact**: scale 22.177, shape 2.119, OFV 640.261); Gompertz fixed-effects
  RCT recovery (alpha/gamma/**loghr** ≈ exact, exercises the `[event_model]` covariate path) + frailty SSE.
  `slow-tests.yml` now passes `survival` so these run nightly.
- ✅ Reference comparison — datasets regenerated (canonical `tte_exp.csv` 100-subj, `tte_weibull.csv`
  100-subj, `tte_gompertz.csv` 300-subj RCT); **ferx FOCEI + base-R `survreg` + nlmixr2 FOCEI columns
  all filled** in each `expected.md` + `docs/estimation/tte.qmd`. ferx ↔ nlmixr2 agree to ~3 digits
  on parameters and (Exp/Weibull) −2LL; **only the NONMEM column is a hand-off** (per-family `README.md`
  + zips — needs a NONMEM licence). nlmixr2 ran locally (macOS gfortran FLIBS workaround). Tracked in **#440**.
- ⚠️ **Finding (#440): FOCEI-Laplace over-estimates frailty ω² on *nonlinear* hazard parameters**
  (Weibull shape +72%, Gompertz gamma +62% at N=2000; does not vanish as ω²→0; SAEM of the same data
  reads ~0.13 vs FOCEI 0.34). Likelihood is exact (fixed-effects matches `survreg`); structural params
  recover. Confirms §3.3/§13 (SAEM/IMP preferred for TTE). FOCEI on a *linear* rate (Exponential) is
  near-unbiased (−7%). **Spun off as #469** (FOCEI nonlinear-frailty ω² reads ~17% above NONMEM/nlmixr2
  on identical data — inner-Hessian accuracy lead, still OPEN). Candidate Phase 3/3b follow-up:
  SAEM/IMP comparison + Shi FD-step audit (§9.3). Validation issue **#440 CLOSED 2026-06-27** — the
  3-tool comparison (ferx FOCEI + `survreg` + nlmixr2 + NONMEM LAPLACIAN) is fully filled and committed;
  residual frailty work lives in #469.

**Parser / DSL:**
- ✅ `[event_model]` expressions can now reference `[individual_parameters]` names — **PR #442 merged
  2026-06-22** (`91da4c37`; hardening `3c2394cb`). `param_fn` now threads the individual-parameters
  evaluator into `parse_event_model_block`; kappa/IOV and NN refs rejected with a clear error.

**ferx-r:**
- ❌ `predict_survival` R wrapper — fully unblocked; ferx-r TTE routing is merged
- ❌ R-side end-to-end TTE test (Tier 2: parse model with `[event_model]`, read CSV,
  verify `obs_records` populated, OFV finite)

### Phase 1b — Competing risks (cause-specific hazard) — ✅ complete

**Scope:** Multiple event types, one `[event_model]` per CMT; a subject experiencing event
type A is right-censored for type B at the same time. Shared random effects link the hazards.
No new infrastructure beyond Phase 1 — this is multiple TTE endpoints with per-CMT routing.

**Status: complete.** Landed across three merged PRs:
- **#494** (merged 2026-06-24) — right-censor TTE simulation at the observation window; the
  prerequisite simulation-censoring fix.
- **#501** (merged 2026-06-25) — competing-risks TTE: earliest-cause event simulation + cause-specific
  cumulative incidence (CIF / Aalen–Johansen). `examples/tte_competing_risks.ferx` + `data/tte_competing_risks.csv`.
- **#526** (merged 2026-06-26, closes #522) — `[simulation] horizon` + TTE-row generation for the
  competing-risks VPC.

**Deliverables (all ✅):**
- ✅ Multiple `[event_model]` blocks keyed by distinct CMT; per-CMT `HazardSpec` (Phase 1 infra #192)
- ✅ Datareader: per-event-type censoring rows (DV=1 on the experienced CMT, DV=0 on the others
  at the same time) — see §3.6 data format
- ✅ **Simulation**: draw each cause's latent event time independently; the observed event is the
  earliest, its CMT is the cause; all other causes censored at that time (§8.8.2) — #501
- ✅ **Prediction**: per-cause cumulative incidence function (CIF) `∫₀ᵗ h_k(u)·S_all(u) du` (#501)
- ✅ Docs + example in `docs/estimation/tte.qmd`; SSE / TTE-row VPC generation (#526)

**Cleanup done:** #531 — review cleanups from #526 (fold sim write-back passes, share
`gather_tte_causes`, reuse `has_tte()`) merged 2026-06-27 via PR #563. Refactor only, no
behaviour change.

**Note (Fine–Gray):** subdistribution-hazard / Fine–Gray CIF modeling is **deferred** (§10.2)
— it needs IPCW weighting and is numerically unstable for sparse data. Cause-specific hazard
covers the standard pharmacometric use case.

### Phase 2 — Joint PK-TTE, ODE hazard accumulator — Slice 2.1 DONE; Slice 2.2 NEXT

**Scope:** Drug-dependent hazard; CHZ as extra ODE state; shared ETA; PK + TTE simultaneously.

**DSL (decided 2026-06-27, shipped in 2.1): auto-append `hazard=`.** User writes `hazard = <expr>`
in `[event_model]` (referencing ODE amounts / theta / eta / covariates); the parser synthesizes and
appends `__chz_<cmt>' = <expr>` (init 0) to `OdeSpec` and records its state index on the endpoint. No
user-written accumulator. Expression compiles in the ODE-RHS namespace (reuses #442 threading).

**Slice 2.1 — fit path ✅ DONE (#564 → PR #567, squash `657800ee`, merged 2026-06-28):**
- `HazardSpec::OdeAccumulated { chz_state }` — note the realized variant carries only `chz_state`
  (the index into the ODE state vector); the hazard is the appended ODE derivative line, not a
  stored closure (no `hazard_fn` field — simpler than the original sketch).
- Parser appends `dCHZ/dt = hazard` to `OdeSpec`; rejects `hazard` + analytic `family`, `hazard`
  without `[odes]`, `hazard` without `cmt`, `hazard` + IOV, and a `__chz_<cmt>` state-name collision.
- Shared seams added: `survival::tte_nll_from_curves(records, cumhaz_at, hazard_at)` (the per-record
  NLL, reused by the analytic path byte-identically) and `survival::ode_cumhaz_hazard(...)` (one ODE
  solve → `H(T)` and `h(T)`); `tte_data_term` ODE branch reads `H(T)`/`h(T)` (Exact + RightCensored).
- FOCEI FD-Hessian re-integrates the ODE per perturbed η; SAEM M-step mirrors it via the same seam.
- `predict_survival` from the ODE; `simulate` of `OdeAccumulated` returns a typed "Slice 2.2" error.
- **Validation: three-way anchor** `tests/reference/pktte_joint/` — ferx ≈ NONMEM ≈ nlmixr2 on the PK
  block (2–3 sig figs); H0/BETA confirmed a flat collinear ridge (corr −0.93 ferx / −0.91 NONMEM);
  OFV-comparability caveat documented. Runnable `examples/pktte_joint.ferx` + Tier-2 `tte_smoke`
  (coverage gate) + Tier-3 `joint_pktte_focei_fit_completes`. (Example is `pktte_joint`, not the
  originally-sketched `pktte_weibull`.)
- **Bug caught & fixed in the same PR:** the NONMEM SE cross-check exposed a pre-existing
  `extract_standard_errors` bug — a theta with a *negative lower bound* (identity-packed, e.g. an
  exposure–hazard slope) had its SE multiplied by the estimate as if log-packed (BETA RSE 5.3% vs
  the correct 20.5%). Now guarded by `theta_packs_log` + a Tier-1 regression test.
- **ferx-r:** pin bump to ferx-core `9fb6cb27` + NEWS = draft PR #208 (bundled `pktte_joint` R
  example/test still to add — local R build is gfortran-blocked, wants CI/toolchain validation).

**Slice 2.2 — simulation (NEXT):**
- **ODE event-location root-finder** `integrate_until_threshold` (§8.8.3) — shared infra
- **Simulation**: drug-driven hazard event-time sampling via the root-finder (lift the typed
  `simulate` guard added in 2.1)
- **Tier-3 SSE** (simulate → fit → recover); Tier-3 convergence already in place from 2.1
- Optionally fold in #570 here (share the Gaussian+TTE ODE solve so the FD-Hessian doesn't
  double-integrate) — perf only, not blocking.

**Slice 2.3 — docs/example polish + comparison table.** (The anchor comparison table already
exists in `tests/reference/pktte_joint/expected.md`; 2.3 surfaces it into `docs/estimation/tte.qmd`.)

**Deferred:**
- **Selective per-state ODE reset** (§8.8.6) → Phase 3 (clock-reset RTTE; no Phase 2 consumer;
  sub-integration fallback available)
- `IntervalCensored`+ODE and left-truncation (`entry_time>0`)+ODE → small follow-ups after 2.1
- **Harden `cif_curves` against NaN cause-rows** in competing-risks `predict_survival` (a failed ODE
  solve at a grid node freezes all-cause survival and zeroes later CIF nodes) — niche; pairs with
  first-classing multi-hazard-ODE competing risks.

### Phase 3 — RTTE

**Scope:** Multiple events per subject; interval likelihood; SAEM primary.

**Deliverables:**
- `type = rtte` in `[event_model]`; `clock = forward | reset` (default `forward`); multiple
  `ObsRecord::Event` per subject
- NLL (§3.3): **clock-forward (default)** `Σ_k log h(t_k) − H(T)` (continuous accumulation);
  **clock-reset** `Σ_k log h(Δ_k) − Σ_k H(Δ_k)` over inter-event gaps via the Phase 2 selective
  reset (§8.8.6)
- **Simulation** (§8.8.4): repeated event sampling to `[simulation] horizon`; for `clock=reset`
  restart the hazard clock per event; final administrative censoring at `horizon`
- SAEM validation vs. nlmixr2 SAEM (primary), NONMEM IMP (secondary)
- Docs: `docs/estimation/rtte.qmd` with estimation method guidance
- Tests: Tier 3 SAEM convergence; **Tier 3 SSE** (simulate RTTE → fit → recover)

### Phase 3b — SAEM proposal option (can happen alongside Phase 3)

**Scope:** Add `saem_proposal = auto | laplace | random_walk` to `[fit_options]`.
`auto` is the new default; existing behaviour (`random_walk`) remains available.

**Impact:** Benefits ALL SAEM runs (Gaussian and non-Gaussian). The acceptance ratio
remains mathematically exact for both proposals — this is a tuning/speed improvement,
not an approximation. Low regression risk: `random_walk` path is unchanged code.

**Deliverables:**
- `SaemProposal` enum in `types.rs`; parsed from `[fit_options]`
- Modify `estimation/saem.rs` E-step: branch on proposal type; track per-subject
  acceptance rate for `auto` mode (exponential moving average over burn-in iterations)
- Fallback: if Laplace Cholesky fails (non-PD Hessian), silently fall back to random-walk
  for that subject
- Benchmark: convergence iterations (warfarin SAEM + RTTE dataset) for all three modes
- Docs: `docs/model-file/fit-options.qmd` — `saem_proposal` entry with guidance table

### Phase 4 — Categorical and Count Models

**Binary:**
- `EndpointLikelihood::Binary`; toenail dataset vs. saemix
- This is mixed-effects **logistic regression** (logit link; probit/cloglog are `LinkFn`
  variants). Include a fixed-effects (`n_eta = 0`) smoke test — ordinary logistic regression
  is the no-random-effect special case (§16 D7)

**Ordinal (proportional odds):**
- `EndpointLikelihood::Ordinal { cuts }`; simulated CDRS-type dataset vs. NONMEM

**Poisson:**
- `EndpointLikelihood::Poisson`; vs. NONMEM F_FLAG=1

**Negative binomial:**
- Extension of Poisson; overdispersion parameter

**Simulation & prediction (all four, §8.8.2):**
- Samplers on the fixed observation grid — `SimOutcome::Category` (binary/ordinal),
  `SimOutcome::Count` (Poisson/NB). These fit the existing grid loop; only the per-draw
  generative step changes.
- `predict()` → `Prediction::CatProbs { probs }` (binary/ordinal) and
  `Prediction::Rate { lambda }` (count)
- **Tier 3 SSE** for each distribution (simulate→fit→recover)

**Files:** `src/types.rs`, `src/stats/likelihood.rs`, `src/parser/model_parser.rs`,
`src/io/datareader.rs`, new `src/categorical/mod.rs`.

### Phase 4b — DTMM

Direct transition probability parameterization; no matrix exponential; vs. NONMEM
Bergstrand 2025 supplementary code.

### Phase 4c — mCTMM (minimal CTMM)

Single-parameter CTMM (`τ = 1/q`, proportional odds steady-state). Stepping stone to
full CTMM. Validates state-observation data reader and CTMM NLL before matrix expm.

### Phase 5 — Time-homogeneous CTMM

**Scope:** Full Q matrix; Padé matrix exponential; Van Loan gradient.

**Deliverables:**
- `src/markov/mod.rs`: `matrix_exp`, `matrix_exp_param_grad`, `ctmm_data_term`
- `[markov_model]` DSL; `type = ctmm`
- **Simulation**: Gillespie/Doob path generator (`src/markov/simulate.rs`); observe the
  simulated path on `[simulation] obs_schedule` (§8.8.2, §8.8.4)
- **Prediction**: state-occupancy vector `π(t) = π₀·expm(Q·t)` → `Prediction::CatProbs`
- Tier 1 unit test: `matrix_exp` vs. series expansion for 2×2 and 3×3 Q
- Tier 3 convergence test: 3-state CTMM vs. R `msm` (CAV dataset)
- **Tier 3 SSE**: simulate path → fit Q → recover rates
- Docs: `docs/estimation/ctmm.qmd` with NONMEM infeasibility rationale

Gate: `#[cfg(feature = "markov")]` initially.

### Phase 6 — Time-inhomogeneous CTMM (drug-driven Q)

**Scope:** Q(t) = f(C(t)); matrix ODE; joint PK-CTMM.

- `ctmm_inhomogeneous_transition` using existing RK45
- `q12 = f(Cc)` DSL expression

### Phase 7 — HMM (Hidden Markov Models)

**Scope:** Latent (unobserved) state sequence; forward algorithm for marginal likelihood.

**Note:** HMM requires marginalization over hidden states — incompatible with single-EBE
BFGS. The inner step becomes EM over hidden states (E-step: forward-backward; M-step:
gradient over θ given expected state occupancies). This is a distinct estimation sub-path.

**Prerequisite:** Phase 5 (CTMM infrastructure) + careful design of inner optimizer dispatch.

### Phase 8 — Custom `[ll_model]` escape hatch

User-specified log-likelihood expression; covers distributions not in built-in list.

---

## 13. Estimation Method Compatibility

| Method | TTE | RTTE | Binary/Ord/Count | CTMM | Notes |
|---|---|---|---|---|---|
| FOCEI (Laplace) | ✓ | ✓ (warn low rates) | ✓ | ✓ | FD Hessian + log-det term |
| FOCE (standard) | ✗ biased | ✗ | ✗ | ✗ | Drops log-det term; incorrect |
| SAEM | ✓ | ✓ **preferred** | ✓ **preferred** | ✓ | f-SAEM proposal (Phase 3b) |
| IMP | ✓ | ✓ preferred | ✓ | ✓ | Auto-extends; ISAMPLE ≥ 300 for NGs |
| GN (pure) | ✗ | ✗ | ✗ | ✗ | Gaussian J'R⁻¹J structure |
| GN-Hybrid | ✓ | ✓ | ✓ | Unlikely | BHHH polish only; the pure-GN warm-start phase is Gaussian-specific and must be **skipped** for non-Gaussian (§9.4) |
| NUTS (future) | ✓ exact | ✓ exact | ✓ exact | ✓ exact | 10–100× slower; exact posterior |
| AGQ (optional) | ✓ q≤3 | ✓ q≤3 | ✓ q≤3 | ✓ q≤3 | Better than Laplace for sparse data |

---

## 14. Testing Strategy

### 14.1 TTE — exponential (analytic reference)

100 subjects; λ=0.1; η~N(0,0.25); 30% censoring. vs. NONMEM LAPLACIAN=1 LIKE=1.
Accept: OFV ±0.5; θ ±10%; SE ±20%.

### 14.2 TTE — Weibull + covariates

100 subjects; α=2, λ=0.2; drug effect (1-cpt PK); 30% censoring.
Reference: NONMEM + `flexsurv::flexsurvreg`.

### 14.3 Joint PK-TTE

100 subjects; warfarin-like 1-cpt PK; Emax hazard linked to AUC.
Reference: NONMEM simultaneous $DES.

### 14.4 RTTE

Holford (2013) RTTE tutorial; exponential hazard; ~5 events/subject expected.
Primary: nlmixr2 SAEM. Secondary: NONMEM IMP.
Also run FOCEI and document degradation.

### 14.5 Binary — toenail dataset

de Backer et al. 1998; 294 subjects; binary. vs. saemix (Comets 2026).
Accept: OFV ±1.0; fixed effects ±15%.

### 14.6 Ordinal

200 subjects; 4-category simulated; proportional odds model.
Reference: NONMEM F_FLAG=1 ordinal logit.

### 14.7 Poisson count

150 subjects; Poisson rate = BASE·exp(β·C)·exp(ETA).
Reference: NONMEM F_FLAG=1 log-Poisson. Also check Plan (2011) — LAPLACE bias should be ~1%.

### 14.8 mCTMM

Simulated 3-state mCTMM; single τ parameter; vs. R `msm` restricted model.

### 14.9 CTMM — CAV dataset

CAV (cardiac allograft vasculopathy); 3 states. Available in R `msm`.
Reference: `msm::msm()`. Accept: Q entries ±15%; OFV ±1.0.

### 14.10 Matrix exponential unit test (Tier 1, fast)

Verify `matrix_exp(A)` matches `Σ Aᵏ/k!` (k=0..20) to 1e-10 for:
- 2×2 diagonal Q (analytic solution available)
- 3×3 dense Q
Verify `matrix_exp_param_grad` matches FD gradient to 1e-8.

### 14.11 Simulation-estimation (SSE) — every endpoint (license-free)

For each endpoint type, a Tier 3 test simulates a dataset from known (θ, Ω) using ferx's own
`simulate()`, refits with ferx, and asserts recovery within Monte-Carlo error (θ within 2×SE,
Ω within ~15%). Requires no external license, so it is always runnable in CI nightly. This is
the primary guard on the new generative code paths (§8.8) — fitting tests alone cannot detect
a wrong sampler. Pair each fit test in §14.1–14.9 with its SSE counterpart.

---

## 15. Data Format Design

### 15.1 TTE data (NONMEM-compatible)

```csv
ID,TIME,DV,EVID,CMT,AMT,MDV
1,0,.,1,1,100,1       ; dose
1,24,0,0,2,.,0        ; censored at t=24, CMT=2 (TTE endpoint)
2,0,.,1,1,100,1
2,15,1,0,2,.,0        ; event at t=15
3,10,0,0,2,.,0        ; interval-censored: event between 10 and t=15
3,15,2,0,2,.,0        ; DV=2 = interval-censored right bound
```

`CMT=2` routes to `EndpointLikelihood::TTE` by `[event_model] cmt = 2` declaration.

**With left truncation (delayed entry)** — optional `TENTRY` column (default 0):

```csv
ID,TIME,TENTRY,DV,EVID,CMT,AMT,MDV
1,40,30,1,0,2,.,0     ; entered risk set at 30, event at 40 → uses H(40) − H(30)
2,55,35,0,0,2,.,0     ; entered at 35, censored at 55
```

Competing risks (cause-specific hazard) reuse the same TTE format with one CMT per event
type and per-event-type censoring rows — see §3.6 and Phase 1b.

### 15.2 RTTE data

```csv
ID,TIME,DV,CMT,EVID,MDV
1,5,1,2,0,0    ; event 1
1,12,1,2,0,0   ; event 2
1,24,0,2,0,0   ; censoring time (final row = observation window T)
```

### 15.3 CTMM / mCTMM / DTMM data (no EVID=3 needed)

```csv
ID,TIME,DV,CMT,EVID,MDV
1,0,1,5,0,0    ; state 1 at t=0
1,3.5,2,5,0,0  ; state 2 at t=3.5
1,8.1,2,5,0,0  ; state 2 at t=8.1
1,15,1,5,0,0   ; state 1 at t=15
```

`CMT=5` routes to `EndpointLikelihood::CTMM` by `[markov_model] cmt = 5`.

### 15.4 Ordinal / binary data

```csv
ID,TIME,DV,CMT,EVID,MDV
1,2,3,4,0,0    ; category 3 at t=2 (CMT=4)
1,4,2,4,0,0    ; category 2 at t=4
```

### 15.5 Mixed joint dataset (PK + ordinal + TTE)

```csv
ID,TIME,DV,CMT,EVID,AMT,MDV
1,0,.,1,1,100,1    ; PK dose
1,2,10.5,1,0,.,0   ; PK observation (Gaussian, CMT=1)
1,4,11.2,1,0,.,0
1,2,2,4,0,.,0      ; ordinal score, CMT=4
1,4,1,4,0,.,0
1,24,1,2,0,.,0     ; TTE event, CMT=2
```

---

## 16. Open Design Decisions

### D1: ObsRecord vs. parallel Vec fields

`Vec<ObsRecord>` (polymorphic) vs. separate `Vec<TteRecord>`, `Vec<DiscreteRecord>`, etc.
Recommendation: Polymorphic `Vec<ObsRecord>` — cleaner iteration, no silent mismatch
between parallel vectors, forward-compatible with new observation types.

### D2: Analytic vs. FD Hessian dispatch

Keep analytic Almquist formula for Gaussian endpoints (performance). FD only for
non-Gaussian. A per-CMT dispatch in `foce_subject_nll_interaction` achieves this.

### D3: SAEM sigma update for non-Gaussian

Skip sigma analytic update for subjects with non-Gaussian endpoints; rely on numerical
gradient from outer optimizer for sigma (or: no sigma for TTE/count endpoints, since
these have no residual variance).

### D4: Matrix exponential crate vs. in-house

Implement inline using nalgebra — ~150 lines, no new dependency, full control,
testable against series definition. The `expm` crate on crates.io is unmaintained (2019).

### D5: Feature flag for markov module

`#[cfg(feature = "markov")]` initially; promote to default after CAV dataset validation.
TTE/categorical have no feature flag — they are core NLME functionality. See D8 for the
overall build & gating strategy this fits into.

### D6: AGQ feature flag

`#[cfg(feature = "agq")]`; expose via `[fit_options] n_agq = 3`. Default n_agq = 1 (= Laplace).

### D7: Fixed-effects (no random effects) as a supported special case

Textbook **logistic regression** and **basic parametric PH/AFT** have *no* random effects.
These are the `n_eta = 0` special case of the generalized NLL: with no η the objective reduces
to the plain observation likelihood `Σⱼ data_term` (no eta-prior, no inner loop). The code
already handles `n_eta = 0` on the IMP path (`importance_sampling.rs`: "the marginal
likelihood is just the observation likelihood"); the inner optimizer becomes a no-op and the
outer optimizer fits θ by direct ML.

**Decision:** Treat `n_eta = 0` as a first-class mode for every non-Gaussian endpoint, not
only Gaussian. Each phase that adds an endpoint must include a Tier 2 smoke test that `fit()`
runs with an empty Ω (e.g. fixed-effects logistic on the toenail data; fixed-effects Weibull
PH on the Phase 1 dataset), so the FOCEI/outer path is verified to tolerate an empty
Cholesky/parameterization. No new API — just guard the empty-Ω path and test it.

### D8: Build & gating strategy — runtime dispatch, not a separate build

**Question:** should the non-Gaussian work be a separate build / blanket Cargo feature
(a "`--nn`-style" gate over everything)?

**Decision: No.** Use the rule the codebase already follows — *core modeling variety →
runtime enum; experimental or toolchain-bound extension → Cargo feature*:

| Capability | Gating | Rationale |
|---|---|---|
| TTE / survival / RTTE / PH / logistic / ordinal / count | **Runtime** `EndpointLikelihood` enum, **default build** | Core NLME, no new deps; one binary, model chosen in the `.ferx` file (like `EstimationMethod` / `ErrorSpec` today); ferx-r gets it with zero feature coordination |
| Markov (CTMM/DTMM/mCTMM/HMM) | `#[cfg(feature = "markov")]` → promote to default after CAV validation (D5) | Specialized (matrix exp, Van Loan); opt-in while maturing — the existing `nn` precedent |
| AGQ | `#[cfg(feature = "agq")]` (D6) | Optional accuracy mode |
| autodiff | retired | Enzyme `autodiff` was retired (#367/#381, 2026-06-20); `gradient = ad` now errors |

**Why not a blanket feature:** model selection is a *runtime / model-file* concern, not a
build concern (nobody should pick a binary to fit Weibull vs. logistic); `#[cfg]`-ing enum
variants + every match arm + parser/datareader branch is invasive (the `nn` feature already
has 20+ cfg sites — acceptable for one module, not for the whole non-Gaussian core); and each
feature doubles the `cargo check --features` CI matrix. Keeping the core ungated confines new
matrix entries to `markov`/`agq`.

**Development lifecycle:** in-progress core code *may* live behind a temporary
`#[cfg(feature = "survival")]` gate so half-finished work stays out of released builds, then
flip to default-on once Phase 1 is validated — the same maturation path `nn` is on. End state
for TTE/categorical is **default-on**.

**Naming note:** do **not** name any of this `nn` — that feature name is already taken in
`Cargo.toml` for neural-network covariate models (DCM) and low-dim NODE dynamics
(`plans/dcm-and-low-dim-node.md`). Use `survival` / `markov` / `agq`.

---

## 17. Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| TTE Laplace FD Hessian numerically unstable | Medium | Shi step-size (§9.3); sentinel return |
| ODE hazard accumulator fails at extreme η | Medium | Existing `1e20` sentinel; floor on h |
| Inner BFGS saddle point for TTE/ordinal | Medium | Saddle detection + escape (§9.5) |
| RTTE Laplace bias at low event rates | High | Default SAEM; document Laplace limitation |
| Standalone single-event TTE frailty ω non-identified | Medium | Prefer fixed-effects hazard; warn + check ω profile/SE; frailty belongs in joint PK-TTE / RTTE (§10.1) |
| Left truncation ignored → upward-biased hazard | Medium | `TENTRY` column; `H(T)−H(T_entry)` correction; validate vs. `survival::survreg(Surv(TENTRY,TIME,event))` (§3.6) |
| CTMM matrix expm unstable for large Q | High | Clamp entries; Padé rescaling built-in |
| NONMEM OFV formula mismatch | High | Validate Exponential TTE vs. NONMEM first |
| Ordinal cut-points not identifiable | Medium | Fix one cut at 0 or constrain parameterization |
| f-SAEM Hessian approximation too rough for proposal | Low | Fallback to random-walk MH if proposal acceptance < 0.01 |
| CTMM with > 5 states too slow | Medium | Krylov subspace for S > 10 (Phase 5+) |
| HMM inner optimizer incompatible with BFGS | High | Separate inner EM sub-path; defer to Phase 7 |
| ODE event-location misses a crossing (steep hazard) | Medium | Dense-output Hermite root-find within the accepted step; cap step size near threshold |
| Gillespie simulation slow at high transition rates | Low | Inherent; cap events/subject and warn; Krylov not needed for simulation |
| TTE simulation never reaches an event (very low hazard) | Low | Administrative censoring at horizon — correct behaviour, not a bug |
| `SimulationResult`/`Prediction` enum change breaks ferx-r glue | Medium | Coordinate the result-type change in the ferx-r plan (§11 step 9) before merge |

---

## 18. Milestone Order

1. **Phase 1** ✅ — Parametric TTE, Laplace (incl. left truncation / delayed entry)  
   All new infrastructure. Lowest risk. Validates FD Hessian + log-det term. (#190/#192/#206/#441/#442)

2. **Phase 1b** ✅ — Competing risks (cause-specific hazard)  
   No new infrastructure; multiple TTE endpoints. (#494/#501/#526; cleanup #531 via #563)

3. **Phase 2** — Joint PK-TTE, ODE hazard accumulator ← **IN PROGRESS** (Slice 2.1 fit path ✅ #564/#567; Slice 2.2 simulation NEXT)  
   Most clinically demanded. Extends Phase 1 via ODE.

4. **Phase 3** — RTTE + **Phase 3b** SAEM proposal option  
   RTTE validates non-Gaussian SAEM. `saem_proposal` option improves all SAEM globally.

5. **Phase 4** — Binary + Ordinal + Poisson + NB  
   Validates generalized NLL for well-understood distributions. Low risk.

6. **Phase 4b** — DTMM  
   Stepping stone; validates state-observation data reader.

7. **Phase 4c** — mCTMM  
   Stepping stone; validates CTMM NLL math before matrix expm.

8. **Phase 5** — Time-homogeneous CTMM  
   Matrix expm + Van Loan gradient. Validate vs. R `msm`.

9. **Phase 6** — Time-inhomogeneous CTMM  
   Drug-driven Q; matrix ODE.

10. **Phase 7** — HMM  
   Forward algorithm; distinct inner optimizer path.

11. **Phase 8** — Custom `[ll_model]` escape hatch  
    Arbitrary user-defined log-likelihood.
