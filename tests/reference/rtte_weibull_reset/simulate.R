# simulate.R — clock-reset (gap-time / renewal) Weibull RTTE with subject frailty.
#
# Phase 3 Slice 3.2 anchor. The hazard clock RESETS at each event, so a subject's
# inter-event gaps are i.i.d. Weibull(scale_i, shape) draws:
#   scale_i = TVSCALE * exp(eta_i),  eta_i ~ N(0, omega2),  shape fixed.
# A Weibull shape != 1 gives a time-varying within-gap hazard, so clock-reset and
# clock-forward genuinely differ here (unlike the memoryless exponential case).
#
# ferx Weibull parameterisation: H(t) = (t/scale)^shape, so a gap is drawn as
# rweibull(shape, scale) (R uses the same scale/shape convention).
#
# Run:  Rscript simulate.R

set.seed(32)

N       <- 100L
TVSCALE <- 5.0    # characteristic inter-event gap
SHAPE   <- 1.5    # > 1: hazard increases within each gap
omega2  <- 0.09   # frailty variance (var of log scale)
horizon <- 30.0   # administrative right-censoring time

rows <- list()
k <- 0L
for (i in seq_len(N)) {
  eta     <- rnorm(1, 0, sqrt(omega2))
  scale_i <- TVSCALE * exp(eta)
  t <- 0
  repeat {
    t <- t + rweibull(1, shape = SHAPE, scale = scale_i)
    if (t >= horizon) break
    k <- k + 1L
    rows[[k]] <- data.frame(ID = i, TIME = round(t, 4), DV = 1L, EVID = 0L, CMT = 2L, MDV = 0L)
  }
  k <- k + 1L
  rows[[k]] <- data.frame(ID = i, TIME = horizon, DV = 0L, EVID = 0L, CMT = 2L, MDV = 0L)
}

dat <- do.call(rbind, rows)
write.csv(dat, "rtte_weibull_reset.csv", row.names = FALSE, quote = FALSE)

cat(sprintf("subjects=%d  events=%d  mean events/subj=%.3f\n",
            N, sum(dat$DV == 1L), sum(dat$DV == 1L) / N))
