# Gauss-Newton (BHHH)

> **Maturity: beta** — see [Feature Maturity](../maturity.md) for what this means.

The Gauss-Newton optimizer is an alternative to the standard FOCE/FOCEI approach that exploits the nonlinear-least-squares structure of the FOCE objective. It uses the BHHH (Berndt-Hall-Hall-Hausman) approximation to the Hessian and typically converges in 10-30 iterations, compared to 100+ evaluations for gradient-based methods like SLSQP.

This approach mirrors NONMEM's modified Gauss-Newton algorithm.

## Variants

| Method | Key | Description |
|--------|-----|-------------|
| Pure Gauss-Newton | `gn` | Fast convergence for well-conditioned problems |
| GN + FOCEI hybrid | `gn_hybrid` | Runs GN first, then polishes with FOCEI for robustness |

## Algorithm Overview

### BHHH Hessian Approximation

The FOCE objective has the form:

\\[ \text{OFV} = 2 \sum_{i=1}^{N} \text{NLL}_i \\]

The BHHH approximation uses the outer product of per-subject gradients as an approximate Hessian:

\\[ H_{\text{BHHH}} = 4 \sum_{i=1}^{N} g_i \, g_i^T \\]

where \\( g_i = \nabla_x \text{NLL}_i \\) is the gradient of the negative log-likelihood for subject \\( i \\) with respect to the packed parameter vector. Per-subject gradients are computed via central finite differences.

This approximation is equivalent to the Gauss-Newton Hessian for log-likelihood objectives and is guaranteed positive semi-definite.

### Levenberg-Marquardt Damping

To improve stability away from the minimum, the Hessian is regularized:

\\[ H_{\text{LM}} = H_{\text{BHHH}} + \lambda \, \text{diag}(H_{\text{BHHH}}) \\]

The damping factor \\( \lambda \\) is adapted during optimization:
- **On successful step** (OFV decreases): \\( \lambda \leftarrow 0.3 \, \lambda \\) (minimum \\( 10^{-6} \\))
- **On rejected step** (OFV increases): \\( \lambda \leftarrow 10 \, \lambda \\)
- If \\( \lambda > 10^6 \\), the algorithm stops

The Newton step is obtained by solving:

\\[ H_{\text{LM}} \, \delta = -\nabla \text{OFV} \\]

via Cholesky decomposition. If the Cholesky fails (singular Hessian), \\( \lambda \\) is increased and the iteration is retried.

### Line Search

Each step uses backtracking line search (up to 15 halvings). At each trial point, EBEs are re-estimated via the inner loop (warm-started from the previous EBEs), ensuring the FOCE objective is evaluated consistently.

### Convergence

The algorithm converges when:
- Relative OFV change \\( < 10^{-6} \\) for at least 3 iterations, or
- Maximum iterations are reached

## Hybrid Mode (GN + FOCEI)

When `method = gn_hybrid`, the algorithm runs in two phases:

1. **GN phase**: Run Gauss-Newton iterations to quickly find the basin of the minimum
2. **FOCEI polish**: Run up to 100 iterations of standard FOCEI optimization, warm-started from the GN result. The polish stage uses the configured outer optimizer (`optimizer` in `[fit_options]`), which defaults to BOBYQA — see [Outer Optimizers](optimizers.md) for the applicability matrix.

The final result is whichever phase produced the lower OFV. This combines the fast convergence of Gauss-Newton with the refined accuracy of FOCEI.

## Configuration

```
[fit_options]
  method     = gn          # or gn_hybrid
  maxiter    = 100         # max GN iterations
  covariance = true
  gn_lambda  = 0.01        # initial Levenberg-Marquardt damping (default 0.01)
```

The initial Levenberg-Marquardt damping factor defaults to 0.01 and adapts automatically. Set `gn_lambda` explicitly (range ~0.001–0.1) if the default causes early step rejection on ill-conditioned models. Setting `gn_lambda` on a non-GN method (e.g. `foce`) emits a warning and is ignored.

## When to Use

**Use `gn`** when:
- You want fast convergence and the model is well-conditioned
- You are iterating on model development and need quick feedback

**Use `gn_hybrid`** when:
- You want robustness against local minima
- The model is complex (many parameters or random effects)
- You want the speed of GN with a safety net

**Stick with `foce`/`focei`** when:
- The model has convergence issues with GN (rare)
- You need exact compatibility with standard FOCE/FOCEI results
