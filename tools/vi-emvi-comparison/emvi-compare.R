#!/usr/bin/env Rscript
#
# nlmixr2 side of the VI cross-implementation comparison (VI_VALIDATION.md Anchor B).
#
# Runs three fits on warfarin and writes a machine-readable results file:
#
#   1. FOCEI          -- calibrates the additive-constant convention against ferx's FOCEI.
#                        This is the step that makes the ELBO/OFV numbers comparable at all;
#                        see VI_VALIDATION.md 4.4. Measured offset so far: zero.
#   2. emvi           -- the variational-EM fit. Population estimates for Tier 1.
#   3. emvi, returnVi -- the same fit returning the raw variational object, so the
#                        per-subject q (mu, packed Cholesky) is reachable for Tier 2.
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
vi_control <- function(...) {
  emviControl(
    seed = 42L, iters = as.integer(Sys.getenv("FERX_VI_ITERS", "2000")),
    nMc = 8L, viFamily = "fullRank",
    optim = "adam", adaptEta = FALSE, perNoCor = 0, klWarmup = 0L, tol = 0,
    likelihood = "focei", covMethod = "", print = 0L, ...
  )
}

fmt <- function(x) sprintf("%.6f", x)
res <- list()

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
res$emvi <- list(
  objf = e$objf,
  est = setNames(e$parFixedDf$`Back-transformed`, rownames(e$parFixedDf)),
  omega = diag(e$omega),
  elbo_last = utils::tail(e$env$viElbo, 1),
  elbo_n = length(e$env$viElbo)
)
cat("\n=== nlmixr2 emvi ===\n")
cat("objf:", fmt(e$objf), "  (likelihood=\"focei\" objective at the emvi estimate)\n")
print(res$emvi$est)
cat("omega diag:", paste(fmt(res$emvi$omega), collapse = "  "), "\n")
cat("ELBO evals:", res$emvi$elbo_n, " last:", fmt(res$emvi$elbo_last), "\n")

# ---- 3. emvi with returnVi: Tier 2 ------------------------------------------------------
vi <- nlmixr2(mod, dat, est = "emvi", control = vi_control(returnVi = TRUE))
mu <- as.matrix(vi$mu) # N x n_eta
lp <- as.matrix(vi$scale) # N x n_eta(n_eta+1)/2, == vi$Lpack under fullRank
d <- ncol(mu)

# The Cholesky diagonal is stored RAW, not logged: src/inner.cpp's grad_L comment reads
# "(-lp)_i eps_j + [i==j] / L_ii", which is d(log|L|)/dL_ii for a raw diagonal.
#
# NOTE (VI_VALIDATION.md 4.7, still open): row- vs column-major packing is not proven.
# Row-major reproduces plausible variances and column-major does not, so row-major is very
# likely right -- but until it is settled, only S[1,1] (which is L[1,1]^2 under EITHER
# convention) should be compared against ferx. The off-diagonals below are provisional.
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
res$meta <- list(
  nlmixr2est = as.character(packageVersion("nlmixr2est")),
  rxode2 = as.character(packageVersion("rxode2")),
  control = as.list(vi_control())[c(
    "seed", "iters", "nMc", "viFamily", "optim",
    "adaptEta", "perNoCor", "klWarmup", "tol", "likelihood"
  )]
)

out <- file.path(outdir, "emvi-results.rds")
saveRDS(res, out)
cat("\nwrote", out, "\n")
cat("compare against ferx: see tools/vi-emvi-comparison/README.md\n")
