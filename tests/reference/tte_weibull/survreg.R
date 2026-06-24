# License-free fixed-effects (no random-effect) Weibull TTE reference.
#
# Uses base-R `survival::survreg`. The validation dataset has between-subject
# variability on the *shape*; this fixed-effects fit collapses that (so its shape
# is an effective/pooled value, not the data-generating 2.0) — it is the exact
# apples-to-apples anchor for a ferx fixed-effects (n_eta = 0) Weibull fit (plan
# D7), NOT a recovery check (use the ferx SSE for recovery of scale=20, shape=2).
#
# Parameterisation bridge (ferx uses H(t) = (t/scale)^shape):
#   survreg fits  log(T) = intercept + scale_sr * W   (W ~ extreme value)
#   => Weibull shape  = 1 / scale_sr           (scale_sr = fit$scale)
#      Weibull scale  = exp(intercept)
#
# Run:  Rscript survreg.R   (from this directory)

library(survival)

dat <- read.csv("tte_weibull.csv")        # ID,TIME,DV,EVID,CMT,MDV ; DV 1=event 0=cens

fit <- survreg(Surv(TIME, DV) ~ 1, data = dat, dist = "weibull")

intercept <- coef(fit)[["(Intercept)"]]
scale_sr  <- fit$scale                     # survreg's "scale" = 1/shape
shape_hat <- 1 / scale_sr
scale_hat <- exp(intercept)
m2ll      <- -2 * as.numeric(logLik(fit))

cat(sprintf("survreg weibull (fixed effects, no omega):\n"))
cat(sprintf("  intercept       = %.6f\n", intercept))
cat(sprintf("  scale_sr (1/shape) = %.6f\n", scale_sr))
cat(sprintf("  -> ferx shape   = %.6f   (pooled/effective fit; NOT the truth -- data-generating shape = 2.0, inflated here because the fixed-effects fit collapses the shape frailty)\n", shape_hat))
cat(sprintf("  -> ferx scale   = %.6f   (pooled/effective fit; data-generating scale = 20.0)\n", scale_hat))
cat(sprintf("  -2 logLik       = %.6f\n", m2ll))
cat(sprintf("  events / N      = %d / %d\n", sum(dat$DV), nrow(dat)))
