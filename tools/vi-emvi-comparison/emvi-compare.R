#!/usr/bin/env Rscript
#
# nlmixr2 side of the VI cross-implementation comparison (VI_VALIDATION.md Anchor B).
#
# Runs two models on warfarin and writes a machine-readable results file.
#
# Part 1 -- DIAGONAL omega (Tier 1/2/3, VI_VALIDATION.md 4.11):
#
#   1. FOCEI          -- calibrates the additive-constant convention against ferx's FOCEI.
#                        This is the step that makes the ELBO/OFV numbers comparable at all;
#                        see VI_VALIDATION.md 4.4. Measured offset so far: zero.
#   2. emvi           -- the variational-EM fit. Population estimates for Tier 1.
#   3. emvi, returnVi -- the same fit returning the raw variational object, so the
#                        per-subject q (mu, packed Cholesky) is reachable for Tier 2.
#
# Part 2 -- MIXED (block + standalone) omega, for the two claims in section 4.9 that no
# other comparator can reach:
#
#   4. FOCEI          -- the correlation VI has to recover, plus the 4.4 convention check
#                        re-run on this model.
#   5. emvi fullRank  -- structural zeros: does emvi hold the undeclared off-diagonals at
#                        zero, as ferx claims both of its omega routes do?
#   6. emvi meanField -- bound looseness: is the diagonal-q ELBO worse than the full-rank
#                        one on a correlated posterior?
#
# Usage:
#   tools/vi-emvi-comparison/run.sh            # from the repo root; sets up the environment
#   Rscript tools/vi-emvi-comparison/emvi-compare.R   # if FERX_RLIB / FLIBS are already set
#
# Requires nlmixr2est >= 7.0 (emvi/fbvi first appear in 7.0). See run.sh and README.md for
# the isolated-library install and the /opt/gfortran linker workaround.

lib <- Sys.getenv("FERX_RLIB")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressMessages(library(nlmixr2est))

stopifnot(
  "nlmixr2est >= 7.0 is required for est=\"emvi\"" =
    packageVersion("nlmixr2est") >= "7.0.0"
)

outdir <- Sys.getenv("FERX_VI_OUT", unset = ".")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

dat <- read.csv(Sys.getenv("FERX_DATA", "data/warfarin.csv"), na.strings = ".")
dat$DV <- as.numeric(dat$DV)
dat$AMT <- as.numeric(dat$AMT)
cat(sprintf("data: %d rows, %d subjects\n", nrow(dat), length(unique(dat$ID))))

# Mirrors tools/vi-emvi-comparison/warfarin_cmp.ferx, which mirrors the tests/vi.rs fixture:
# 1-cpt oral, lognormal eta on CL/V/KA, proportional error.
#
# The mu-referenced form is the same model as ferx's: exp(log(TVCL) + eta) == TVCL * exp(eta),
# which is why the initial estimates below are logs of ferx's. linCmt() rather than explicit
# ODEs so both sides use a closed-form solution and an ODE-vs-analytic difference cannot
# contaminate the comparison.
mod <- function() {
  ini({
    tvcl <- log(0.13)
    tvv <- log(8.0)
    tvka <- log(1.0)
    eta.cl ~ 0.09
    eta.v ~ 0.04
    eta.ka ~ 0.30
    prop.err <- 0.1
  })
  model({
    cl <- exp(tvcl + eta.cl)
    v <- exp(tvv + eta.v)
    ka <- exp(tvka + eta.ka)
    linCmt() ~ prop(prop.err)
  })
}

# emvi's defaults would each confound the comparison; see VI_VALIDATION.md 4.3.
#
#   optim="advi"      the Kucukelbir adaptive step-size rule, not Adam
#   adaptEta=TRUE     a step-size search over etaCandidates for the first min(iters,75) iters
#   perNoCor=0.75     holds declared omega off-diagonals at ZERO for 75% of the run -- on a
#                     block_omega model this alone looks like a correlation disagreement
#   tol               early-stops on relative ELBO change; ferx deliberately has no tolerance
#   nMc=1             ferx's vi_mc_samples defaults to 8
#
# klWarmup=0 is already the default (no prior tempering) and matches ferx having none.
# covMethod="" skips their covariance step: under pointEstimate=TRUE, covMethod="vi" silently
# falls back to FOCEi "r,s", which is not what ferx's FD-of-OFV Hessian computes.
# viFamily is a named formal rather than left to `...`: passing it through `...` collides with
# the hardcoded default ("matched by multiple actual arguments") the moment fit 6 overrides it.
vi_control <- function(viFamily = "fullRank", ...) {
  emviControl(
    # 20000, not 2000. At 2000 `emvi` is NOT converged on this model and the earlier runs of
    # this harness measured that rather than the estimator: sigma was still falling (0.013128 ->
    # 0.010369 by 20000, converging on the FOCEI/AGQ value), the per-subject posteriors were
    # erratic (across-subject spread on eta_CL 23.8x at 2000, 2.9x at 20000), and `tol = 0`
    # below removes the early stop that would otherwise have caught it. See VI_VALIDATION.md
    # 4.11 and its correction note.
    seed = 42L, iters = as.integer(Sys.getenv("FERX_VI_ITERS", "20000")),
    nMc = 8L, viFamily = viFamily,
    optim = "adam", adaptEta = FALSE, perNoCor = 0, klWarmup = 0L, tol = 0,
    likelihood = "focei", covMethod = "", print = 0L, ...
  )
}

fmt <- function(x) sprintf("%.6f", x)
res <- list()

# emvi's ELBO trace is a stochastic sequence sampled every `evalElbo` iterations, so its last
# value carries the full gradient noise of one step -- the same last-iterate effect 4.11
# suspects behind the S[1,1] spread. Report the tail mean alongside it so the family
# comparison in (b) does not rest on a single noisy draw.
elbo_summary <- function(fit) {
  tr <- fit$env$viElbo
  if (is.null(tr) || !length(tr)) {
    return(list(n = 0L, last = NA_real_, tail_mean = NA_real_, trace = numeric(0)))
  }
  k <- min(5L, length(tr))
  list(
    n = length(tr), last = utils::tail(tr, 1),
    tail_mean = mean(utils::tail(tr, k)), tail_k = k, trace = tr
  )
}

# ---- The ELBO convention, derived from nlmixr2est's source (VI_VALIDATION.md 4.7, closed) --
#
# `viElbo` is NOT on ferx's scale, and the difference is a constant, not a mystery. From
# nlmixr2est 7.0.2 `src/inner.cpp` (adviElboGradCoreFR, :15059-15163) the reported value is
#
#     elbo = sum_i [ -likInner0(eta_i)  +  sum_k log|L_kk,i| ]  -  0.5 * N * log|Omega|
#
# averaged over `nMc` draws (:15239). `likInner0` returns the NEGATIVE log joint on the 1x
# scale with NO 2*pi constants (:1738ff, `-0.5*err^2/r - 0.5*log r` per observation and
# `-0.5*eta' Omega^-1 eta`), so relative to a fully-normalized ELBO this drops, per subject:
#
#     the eta-prior normalizer      -(d/2) log 2*pi
#     the Gaussian entropy constant +(d/2) (1 + log 2*pi)   [entropy = sum log|L_kk| + that]
#
# which net to +d/2. Both tools drop the observation constant (1/2) n_obs log 2*pi -- the
# NONMEM convention, and the FOCEI check above measures that agreement rather than assuming
# it. So, at the same parameter vector,
#
#     ferx_ELBO = emvi_ELBO + N*d/2,   and ferx reports -2*ELBO.
#
# On warfarin (N = 10, d = 3) that is +15 exactly.
elbo_to_ferx <- function(elbo, N, d) {
  if (is.null(elbo) || !length(elbo) || is.na(elbo)) {
    return(c(elbo = NA_real_, neg_two_elbo = NA_real_))
  }
  ferx_elbo <- elbo + N * d / 2
  c(elbo = ferx_elbo, neg_two_elbo = -2 * ferx_elbo)
}

# =========================================================================================
# PART 1 -- diagonal omega
# =========================================================================================

# ---- 1. FOCEI: the constant-convention anchor -------------------------------------------
f <- nlmixr2(mod, dat, est = "focei", control = foceiControl(print = 0L))
res$focei <- list(
  objf = f$objf,
  est = setNames(f$parFixedDf$`Back-transformed`, rownames(f$parFixedDf)),
  omega = diag(f$omega)
)
cat("\n=== nlmixr2 FOCEI ===\n")
cat("objf:", fmt(f$objf), "\n")
print(res$focei$est)
cat("omega diag:", paste(fmt(res$focei$omega), collapse = "  "), "\n")

# ---- 2. emvi: Tier 1 --------------------------------------------------------------------
e <- nlmixr2(mod, dat, est = "emvi", control = vi_control())
# elbo_summary() (defined below for Part 2) is used here too: the LAST iterate of a
# stochastic optimizer carries the full noise of one step, and reading the diagonal arm's
# bound off it is what left 4.11's Tier 3 looking like a scale mismatch. Keep the trace.
res$emvi <- list(
  objf = e$objf,
  est = setNames(e$parFixedDf$`Back-transformed`, rownames(e$parFixedDf)),
  omega = diag(e$omega),
  elbo_last = utils::tail(e$env$viElbo, 1),
  elbo_n = length(e$env$viElbo),
  elbo = elbo_summary(e),
  elbo_ferx = elbo_to_ferx(elbo_summary(e)$tail_mean, N = length(unique(dat$ID)), d = 3L)
)
cat("\n=== nlmixr2 emvi ===\n")
cat("objf:", fmt(e$objf), "  (likelihood=\"focei\" objective at the emvi estimate)\n")
print(res$emvi$est)
cat("omega diag:", paste(fmt(res$emvi$omega), collapse = "  "), "\n")
cat("ELBO evals:", res$emvi$elbo_n, " last:", fmt(res$emvi$elbo_last),
  " mean(last", res$emvi$elbo$tail_k, "):", fmt(res$emvi$elbo$tail_mean), "\n")
cat("  in ferx's convention (+N*d/2): ELBO", fmt(res$emvi$elbo_ferx[["elbo"]]),
  " -2*ELBO", fmt(res$emvi$elbo_ferx[["neg_two_elbo"]]),
  " <- compare with ferx's neg_two_elbo\n")

# ---- 3. emvi with returnVi: Tier 2 ------------------------------------------------------
vi <- nlmixr2(mod, dat, est = "emvi", control = vi_control(returnVi = TRUE))
mu <- as.matrix(vi$mu) # N x n_eta
lp <- as.matrix(vi$scale) # N x n_eta(n_eta+1)/2, == vi$Lpack under fullRank
d <- ncol(mu)

# The Cholesky diagonal is stored RAW, not logged: src/inner.cpp's grad_L comment reads
# "(-lp)_i eps_j + [i==j] / L_ii", which is d(log|L|)/dL_ii for a raw diagonal.
#
# ROW-MAJOR, and this is now READ FROM THE SOURCE rather than inferred (VI_VALIDATION.md 4.7,
# closed). nlmixr2est 7.0.2 `src/inner.cpp` indexes the pack as `Lpack(s, i*(i+1)/2 + j)` for
# `j <= i` when it draws `eta = mu + L eps` (:15107), and unpacks it the same way when it
# reports the per-subject covariance: `for r, for c <= r: L(r,c) = Lpack(i, r*(r+1)/2 + c)`
# (:15293). Row `i` starts at offset i(i+1)/2, column `j` within it -- row-major lower
# triangular. The earlier reading (plausible variances under row-major, implausible under
# column-major) was right, and `tier2-offdiag.R` now also confirms it against the Anchor C
# NUTS reference, where column-major misses the posterior correlations by up to 1.39.
unpack_row_major <- function(v) {
  L <- matrix(0, d, d)
  k <- 1
  for (i in seq_len(d)) {
    for (j in seq_len(i)) {
      L[i, j] <- v[k]
      k <- k + 1
    }
  }
  L
}
covs <- lapply(seq_len(nrow(lp)), function(i) {
  L <- unpack_row_major(lp[i, ])
  L %*% t(L)
})

cat("\n=== emvi per-subject q (Tier 2) ===\n")
cat("eta order:", paste(vi$etaNames, collapse = ", "), "\n")
for (i in seq_len(min(3L, nrow(mu)))) {
  cat(sprintf(
    "subj %d  mean: %s   var(diag): %s\n", i,
    paste(fmt(mu[i, ]), collapse = ", "),
    paste(sprintf("%.8f", diag(covs[[i]])), collapse = ", ")
  ))
}
cat("(S[1,1] is packing-independent; off-diagonals are provisional -- see above)\n")

res$q <- list(mu = mu, packed = lp, cov = covs, eta_names = vi$etaNames)
# =========================================================================================
# PART 2 -- mixed omega: block (ETA_CL, ETA_V) + standalone ETA_KA
# =========================================================================================
#
# Reaches the two claims in VI_VALIDATION.md 4.9 that nothing else can:
#
#   (a) structural zeros. docs/estimation/vi.qmd promises that in a mixed omega the
#       UNDECLARED off-diagonals stay "at exactly zero rather than letting them pick up
#       sampling correlation", for both vi_omega_update routes. `fbvi` errors outright on a
#       correlated block, so emvi is the only external VI that estimates one at all.
#   (b) mean_field vs full_rank bound looseness, since the families match by name.
#
# perNoCor = 0 is load-bearing here in a way it was not in Part 1: the default 0.75 pins
# DECLARED off-diagonals at zero for three quarters of the run, which on this model would
# look exactly like a correlation disagreement (VI_VALIDATION.md 4.3).
#
# Mirrors warfarin_block_cmp.ferx / vi_block_*.ferx, and the omega is the same shape as
# src/estimation/vi/run_tests.rs::mixed_omega_fixture -- ETA_KA outside the block, so
# omega(KA,CL) and omega(KA,V) are the structural zeros under test.
mod_block <- function() {
  ini({
    tvcl <- log(0.13)
    tvv <- log(8.0)
    tvka <- log(1.0)
    eta.cl + eta.v ~ c(
      0.09,
      0.02, 0.04
    )
    eta.ka ~ 0.30
    prop.err <- 0.1
  })
  model({
    cl <- exp(tvcl + eta.cl)
    v <- exp(tvv + eta.v)
    ka <- exp(tvka + eta.ka)
    linCmt() ~ prop(prop.err)
  })
}

# Pull omega by ETA NAME rather than by position: the block declaration fixes the order here,
# but a silent reorder would otherwise turn into a wrong "structural zero" verdict.
omega_at <- function(om, a, b) {
  nm <- dimnames(om)[[1]]
  stopifnot(a %in% nm, b %in% nm)
  om[match(a, nm), match(b, nm)]
}

# The claim is EXACT zero, not small: ferx reconstructs omega[i,j] = sum_k L[i,k] L[j,k] over
# a Cholesky whose structural-zero slots carry no gradient, so its own regression test asserts
# bit equality (run_tests.rs:564). emvi is a different implementation and gets a tolerance --
# but a tight one, since anything above it means the slot is being estimated, not held.
STRUCTURAL_ZERO_TOL <- 1e-10
report_block <- function(label, fit, elbo = NULL) {
  om <- fit$omega
  zeros <- list(
    c("eta.ka", "eta.cl"),
    c("eta.ka", "eta.v")
  )
  vals <- vapply(zeros, function(z) omega_at(om, z[1], z[2]), numeric(1))
  held <- all(abs(vals) <= STRUCTURAL_ZERO_TOL)
  cat(sprintf("\n=== nlmixr2 %s (mixed omega) ===\n", label))
  cat("objf:", fmt(fit$objf), "\n")
  print(setNames(fit$parFixedDf$`Back-transformed`, rownames(fit$parFixedDf)))
  cat("omega diag:", paste(fmt(diag(om)), collapse = "  "), "\n")
  cat("omega block cov(CL,V):", fmt(omega_at(om, "eta.v", "eta.cl")), "\n")
  cat(sprintf(
    "structural zeros cov(KA,CL) / cov(KA,V): %s / %s  -> %s\n",
    format(vals[1], scientific = TRUE, digits = 3),
    format(vals[2], scientific = TRUE, digits = 3),
    if (held) "HELD" else "NOT HELD"
  ))
  if (!is.null(elbo)) {
    cat(sprintf(
      "ELBO evals: %d  last: %s  mean(last %d): %s\n",
      elbo$n, fmt(elbo$last), elbo$tail_k, fmt(elbo$tail_mean)
    ))
  }
  list(
    objf = fit$objf,
    est = setNames(fit$parFixedDf$`Back-transformed`, rownames(fit$parFixedDf)),
    omega = om,
    block_cov = omega_at(om, "eta.v", "eta.cl"),
    structural_zeros = setNames(vals, c("ka_cl", "ka_v")),
    structural_zeros_held = held,
    elbo = elbo
  )
}

# ---- 4. FOCEI on the mixed omega: the correlation VI has to recover ----------------------
#
# ASYMMETRY WARNING. nlmixr2 honours the declared block structure here, but ferx's FOCEI does
# NOT -- measured 2026-08-20 it estimates the full lower triangle on this same declaration
# (n_parameters = 10 rather than 8), so its cov(KA,CL) / cov(KA,V) come back non-zero. ferx's
# VI arms DO hold them at zero. So this FOCEI pair is a valid cross-tool check of the block's
# own cov(CL,V) and of the 4.4 constant convention, but an omega comparison across the KA row
# is comparing two different models. See warfarin_block_cmp.ferx.
fb <- nlmixr2(mod_block, dat, est = "focei", control = foceiControl(print = 0L))
res$block_focei <- report_block("FOCEI", fb)

# ---- 5. emvi, fullRank: claim (a), and the baseline for claim (b) ------------------------
eb <- nlmixr2(mod_block, dat, est = "emvi", control = vi_control())
res$block_emvi_full_rank <- report_block("emvi fullRank", eb, elbo_summary(eb))

# ---- 6. emvi, meanField: claim (b) -------------------------------------------------------
# Deliberately NOT run with returnVi. Under meanField `scale` is diagonal-only and its layout
# (raw sd? log sd?) is not established, and 4.7's row-vs-column-major question is still open
# for the fullRank packing -- so a per-subject block-model q would be unpackable on two counts.
# Claim (b) is population-level and needs none of it.
em <- nlmixr2(mod_block, dat, est = "emvi", control = vi_control(viFamily = "meanField"))
res$block_emvi_mean_field <- report_block("emvi meanField", em, elbo_summary(em))

# ---- claim (b), and why this arm can only be directional ---------------------------------
#
# One thing does work in this comparison's favour: meanField-vs-fullRank is an ORDERING WITHIN
# one implementation, so the unreconciled ELBO scale from 4.11 Tier 3 cancels out -- whatever
# constant or per-something normalisation separates emvi's ELBO from ferx's is the same in
# both arms.
#
# What does NOT cancel is the parameter vector. Each arm runs its own M-step, so the two ELBOs
# are evaluated at different (theta, Omega, sigma) -- and the ELBO is a bound on -2 log L AT
# the parameter vector where it was evaluated. On the ferx side the same two arms land 14.5%
# apart in sigma (see vi_block_mean_field.ferx), which is far too large to treat the ELBO
# difference as a pure family effect. So this reports the ELBO gap AND the parameter gap
# beside it, and refuses a verdict when the second is not small.
#
# The clean measurement is both families' ELBO at ONE parameter vector. That is a Tier-1 unit
# test over elbo_agq_bound.rs (family_tests.rs already loops the families over a fixed
# fixture), not something this harness can reach.
fr <- res$block_emvi_full_rank
mf <- res$block_emvi_mean_field
par_rel <- function(a, b) {
  nm <- intersect(names(a), names(b))
  max(abs((b[nm] - a[nm]) / pmax(abs(a[nm]), 1e-12)))
}
drift <- par_rel(fr$est, mf$est)
sig_drift <- max(abs((diag(mf$omega) - diag(fr$omega)) / pmax(abs(diag(fr$omega)), 1e-12)))

cat("\n=== claim (b): mean_field vs full_rank bound, within emvi ===\n")
cat(sprintf(
  "fullRank ELBO %s   meanField ELBO %s   (tail means; higher ELBO = tighter bound)\n",
  fmt(fr$elbo$tail_mean), fmt(mf$elbo$tail_mean)
))
cat(sprintf(
  "parameter drift between the arms: %.2f%% (fixed effects), %.2f%% (omega diagonal)\n",
  100 * drift, 100 * sig_drift
))
CONFOUND_TOL <- 0.02
if (max(drift, sig_drift) > CONFOUND_TOL) {
  cat(sprintf(
    "CONFOUNDED: the arms converged >%.0f%% apart, so the ELBO gap mixes the family effect\n",
    100 * CONFOUND_TOL
  ))
  cat("  with the parameter difference. Direction only, not a measurement of looseness.\n")
} else {
  cat("arms agree on parameters to within tolerance, so the ELBO gap is attributable\n")
  cat("  to the variational family.\n")
}
cat(sprintf(
  "direction: meanField is %s, which is %s with vi.qmd's claim that a diagonal q\n  is looser on a correlated posterior\n",
  if (isTRUE(mf$elbo$tail_mean < fr$elbo$tail_mean)) "LOWER" else "NOT lower",
  if (isTRUE(mf$elbo$tail_mean < fr$elbo$tail_mean)) "consistent" else "INCONSISTENT"
))
res$claim_b <- list(
  elbo_full_rank = fr$elbo$tail_mean, elbo_mean_field = mf$elbo$tail_mean,
  param_drift = drift, omega_drift = sig_drift,
  confounded = max(drift, sig_drift) > CONFOUND_TOL
)

res$meta <- list(
  nlmixr2est = as.character(packageVersion("nlmixr2est")),
  rxode2 = as.character(packageVersion("rxode2")),
  # viFamily is the one control that varies across fits (fit 6 overrides it), so it is
  # recorded per-fit above as well; everything else here is shared by all emvi arms.
  control = as.list(vi_control())[c(
    "seed", "iters", "nMc", "viFamily", "optim",
    "adaptEta", "perNoCor", "klWarmup", "tol", "likelihood"
  )],
  structural_zero_tol = STRUCTURAL_ZERO_TOL
)

out <- file.path(outdir, "emvi-results.rds")
saveRDS(res, out)
cat("\nwrote", out, "\n")
cat("compare against ferx: see tools/vi-emvi-comparison/README.md\n")
