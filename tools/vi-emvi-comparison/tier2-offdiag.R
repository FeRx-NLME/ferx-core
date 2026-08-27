#!/usr/bin/env Rscript
#
# Tier 2b, off-diagonals: the per-subject variational COVARIANCE, ferx and `emvi` both read
# against the Anchor C NUTS reference (VI_VALIDATION.md 4.14 / 5.1).
#
# Why this is a separate script from emvi-compare.R: it needs output from BOTH harnesses --
# `emvi-results.rds` (this directory's run.sh) and `anchor-c.json`
# (tools/vi-nuts-anchor/run.sh) -- and neither should have to run for the other to be useful.
#
# Usage, after both harnesses have run:
#     Rscript tools/vi-emvi-comparison/tier2-offdiag.R
#
# Reads $FERX_VI_OUT (default $FERX_VI_STATE/results).
#
# ---------------------------------------------------------------------------------------
# WHAT THIS SETTLES, AND WHY IT NEEDED AN ARBITER
#
# 4.11 compared per-subject variances ferx-against-emvi and got the answer backwards, because
# neither tool is ground truth (4.11a). The off-diagonals were left out entirely on top of
# that, pending the row- vs column-major question about nlmixr2's `Lpack`.
#
# Both blockers are now gone. The packing is row-major, read from nlmixr2est 7.0.2's own
# source (`src/inner.cpp:15107` draws with `Lpack(s, i*(i+1)/2 + j)`, `:15293` unpacks the
# same way), so the off-diagonals are addressable; and Anchor C's NUTS run supplies a
# reference posterior covariance -- the whole matrix, not just the diagonal -- that neither
# implementation computes.
#
# The script therefore does two things:
#
#   1. Confirms the packing against the arbiter, by unpacking BOTH ways and showing which
#      one reproduces the reference correlations. This is a measurement, not the earlier
#      plausibility argument.
#   2. Reports each tool's correlations against NUTS, which is the Tier-2b statement that
#      4.14 had to leave open.
#
# CAVEAT to read before the numbers. Anchor C fixes every population parameter at the AGQ
# estimate, so its NUTS posterior conditions on THAT parameter vector. ferx's `vi_cov` in
# `anchor-c.json` was produced at the same fixed vector; `emvi`'s q comes from its own fit, so
# its sigma differs by ~8% and posterior width scales with sigma^2. Correlations are the
# comparable quantity across that difference -- variances are shown, but the emvi column
# carries that confound and the ratios should not be read as an accuracy claim.
# ---------------------------------------------------------------------------------------

lib <- Sys.getenv("FERX_RLIB")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))

state <- Sys.getenv("FERX_VI_STATE", unset = path.expand("~/.local/share/ferx-vi-validation"))
outdir <- Sys.getenv("FERX_VI_OUT", unset = file.path(state, "results"))

emvi_path <- file.path(outdir, "emvi-results.rds")
anchor_path <- file.path(outdir, "anchor-c.json")
for (p in c(emvi_path, anchor_path)) {
  if (!file.exists(p)) {
    stop(sprintf(
      "%s is missing. Run tools/vi-emvi-comparison/run.sh (emvi-results.rds) and\n  %s",
      p, "tools/vi-nuts-anchor/run.sh (anchor-c.json) first."
    ), call. = FALSE)
  }
}

res <- readRDS(emvi_path)
ac <- jsonlite::fromJSON(anchor_path)

lp <- as.matrix(res$q$packed)
mu <- as.matrix(res$q$mu)
d <- ncol(mu)
N <- nrow(lp)
stopifnot(
  "emvi q and the NUTS reference must cover the same subjects" = N == length(ac$ids),
  "the packed Cholesky must hold d(d+1)/2 entries" = ncol(lp) == d * (d + 1) / 2
)

# Row-major is the production convention (see the header). Column-major is kept only so the
# comparison below can show what it costs.
unpack <- function(v, major = c("row", "col")) {
  L <- matrix(0, d, d)
  k <- 1
  if (match.arg(major) == "row") {
    for (i in seq_len(d)) for (j in seq_len(i)) {
      L[i, j] <- v[k]
      k <- k + 1
    }
  } else {
    for (j in seq_len(d)) for (i in j:d) {
      L[i, j] <- v[k]
      k <- k + 1
    }
  }
  L
}

# The three off-diagonal correlations of a 3x3, in (1,2) (1,3) (2,3) order.
corrs <- function(S) {
  s <- sqrt(diag(S))
  c(S[2, 1] / (s[1] * s[2]), S[3, 1] / (s[1] * s[3]), S[3, 2] / (s[2] * s[3]))
}

pair_names <- c("(CL,V)", "(CL,KA)", "(V,KA)")
fmt <- function(x) sprintf("%7.3f", x)

emvi_cov <- function(i, major) {
  L <- unpack(lp[i, ], major)
  L %*% t(L)
}

nuts_c <- t(sapply(seq_len(N), function(i) corrs(ac$nuts_cov[i, , ])))
ferx_c <- t(sapply(seq_len(N), function(i) corrs(ac$vi_cov[i, , ])))

cat("eta order:", paste(res$q$eta_names, collapse = ", "), "\n")
cat("subjects:", N, "  NUTS reference: anchor-c.json (Anchor C, population parameters FIXed)\n")
cat("\n=== 1. the packing convention, against the arbiter ===\n")
cat("        pair          ", paste(sprintf("%9s", pair_names), collapse = ""), "\n")
verdicts <- list()
for (mj in c("row", "col")) {
  ec <- t(sapply(seq_len(N), function(i) corrs(emvi_cov(i, mj))))
  worst <- apply(abs(ec - nuts_c), 2, max)
  verdicts[[mj]] <- max(worst)
  cat(sprintf(
    "%-6s median corr    %s\n       worst |diff| NUTS %s\n",
    mj, paste(fmt(apply(ec, 2, median)), collapse = "  "),
    paste(fmt(worst), collapse = "  ")
  ))
}
cat("NUTS   median corr    ", paste(fmt(apply(nuts_c, 2, median)), collapse = "  "), "\n")
cat(sprintf(
  "\nverdict: %s-major reproduces the reference correlations (worst |diff| %.3f against\n",
  if (verdicts$row < verdicts$col) "row" else "column",
  min(verdicts$row, verdicts$col)
))
cat(sprintf(
  "  %.3f the other way). This agrees with nlmixr2est's source, which is the primary\n  evidence; this is the independent check.\n",
  max(verdicts$row, verdicts$col)
))

cat("\n=== 2. Tier 2b: each tool's per-subject q against NUTS ===\n")
ec <- t(sapply(seq_len(N), function(i) corrs(emvi_cov(i, "row"))))
cat("                     ", paste(sprintf("%9s", pair_names), collapse = ""), "\n")
cat("median corr  NUTS    ", paste(fmt(apply(nuts_c, 2, median)), collapse = "  "), "\n")
cat("median corr  ferx VI ", paste(fmt(apply(ferx_c, 2, median)), collapse = "  "), "\n")
cat("median corr  emvi    ", paste(fmt(apply(ec, 2, median)), collapse = "  "), "\n")
cat("worst |diff| ferx VI ", paste(fmt(apply(abs(ferx_c - nuts_c), 2, max)), collapse = "  "), "\n")
cat("worst |diff| emvi    ", paste(fmt(apply(abs(ec - nuts_c), 2, max)), collapse = "  "), "\n")

var_ratio <- function(get) {
  t(sapply(seq_len(N), function(i) diag(get(i)) / diag(ac$nuts_cov[i, , ])))
}
cat("\nvariance ratios against NUTS (median over subjects; emvi carries the sigma confound\n")
cat("  in the header -- do not read it as an accuracy claim):\n")
cat("  ferx VI / NUTS ", paste(fmt(apply(var_ratio(function(i) ac$vi_cov[i, , ]), 2, median)), collapse = "  "), "\n")
cat("  emvi    / NUTS ", paste(fmt(apply(var_ratio(function(i) emvi_cov(i, "row")), 2, median)), collapse = "  "), "\n")

out <- file.path(outdir, "tier2-offdiag.rds")
saveRDS(list(
  pair_names = pair_names, nuts = nuts_c, ferx = ferx_c, emvi = ec,
  packing_worst = unlist(verdicts)
), out)
cat("\nwrote", out, "\n")
cat("interpretation: VI_VALIDATION.md section 4.14 (Tier 2b) and 4.7 (the packing question)\n")
