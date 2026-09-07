#!/usr/bin/env Rscript
#
# Generator for the GAM-screening anchor fixture (#1114).
#
# Produces two things:
#
#   1. crates/ferx-tools/tests/data/gam_anchor_ebes.csv — the 60-subject EBE /
#      covariate table the Rust test reads with include_str!.
#   2. the reference ΔAIC values asserted in
#      crates/ferx-tools/src/gam.rs::tests::xpose4_anchor_delta_aic_matches_reference
#
# Run from the repository root:
#
#   Rscript docs/gam_anchor_reference/gen_anchor.R
#
# Requires only base R plus `splines` (both ship with R). `gam::gam()` is the
# engine xpose4::xpose.gam() uses, but for a single-covariate Gaussian
# regression it reduces exactly to lm(), and the constant n(1 + log 2π) in the
# usual AIC cancels in a ΔAIC — so lm.fit() under the screen's own AIC formula
# reproduces xpose4's numbers to floating point.

set.seed(20240101)

n <- 60

# ── Covariates ───────────────────────────────────────────────────────────────
WT   <- round(rnorm(n, mean = 70, sd = 12), 2)
CRCL <- round(rnorm(n, mean = 90, sd = 25), 2)
# `sample()`, not `rbinom()`: both consume the same number of uniforms, so the
# downstream ETA draws are identical either way, but only this one reproduces
# the committed SEX column bit-for-bit.
SEX  <- sample(0:1, n, replace = TRUE)

# ── ETAs ─────────────────────────────────────────────────────────────────────
# ETA_CL is driven by WT; ETA_V by SEX. CRCL is a decoy with no true effect.
ETA_CL <- round(0.40 * (WT - 70) / 70 + rnorm(n, 0, 0.15), 6)
ETA_V  <- round(0.35 * (SEX - 0.5) + rnorm(n, 0, 0.20), 6)

ebes <- data.frame(
  ID = seq_len(n),
  ETA_CL = ETA_CL,
  ETA_V = ETA_V,
  WT = WT,
  CRCL = CRCL,
  SEX = SEX
)

out <- file.path("crates", "ferx-tools", "tests", "data", "gam_anchor_ebes.csv")
write.csv(ebes, out, row.names = FALSE)
cat("wrote", out, "\n\n")

# ── Reference ΔAIC ───────────────────────────────────────────────────────────
# The screen's AIC, verbatim: n·log(RSS/n) + 2p. Not stats::AIC(), which adds
# the constant n(1 + log 2π) + 2 for the estimated variance — that constant
# cancels in ΔAIC, but only if both sides use the same one.
aic <- function(X, y) {
  fit <- lm.fit(X, y)
  rss <- sum(fit$residuals^2)
  length(y) * log(rss / length(y)) + 2 * ncol(X)
}

# Candidate designs, matching gam.rs: linear, natural splines at df = 2 and 3
# (the xpose4 defaults), one-hot categorical with the lowest level as reference.
delta_aic <- function(y, x, kind) {
  n <- length(y)
  null <- aic(cbind(1, rep(0, n))[, 1, drop = FALSE], y)

  if (kind == "categorical") {
    levs <- sort(unique(x))
    if (length(levs) < 2 || length(levs) >= n) return(NA_real_)
    D <- outer(x, levs[-1], function(a, b) as.numeric(a == b))
    return(null - aic(cbind(1, D), y))
  }

  # Centre and scale, as the screen does: an affine reparameterisation of the
  # design's column space leaves RSS and p — and therefore AIC — unchanged, so
  # this only affects conditioning.
  z <- (x - mean(x)) / sqrt(mean((x - mean(x))^2))

  cands <- c(aic(cbind(1, z), y))
  for (df in c(2, 3)) {
    if (n <= df + 1) next
    cands <- c(cands, aic(cbind(1, splines::ns(z, df = df)), y))
  }
  null - min(cands)
}

cat("Reference delta_aic (assert these in gam.rs):\n")
for (eta_name in c("ETA_CL", "ETA_V")) {
  y <- ebes[[eta_name]]
  cat(" ", eta_name, "\n")
  for (cv in c("WT", "CRCL", "SEX")) {
    kind <- if (cv == "SEX") "categorical" else "continuous"
    cat(sprintf("    %-6s %12.6f\n", cv, delta_aic(y, ebes[[cv]], kind)))
  }
}
