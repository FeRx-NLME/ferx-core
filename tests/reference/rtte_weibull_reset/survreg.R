# survreg.R — exact, license-free fixed-effects anchor for clock-reset Weibull RTTE.
#
# Under clock-reset the hazard restarts at each event, so a subject's inter-event gaps
# are independent Weibull observations (the observed gaps are events, the final gap to
# the horizon is right-censored). WITHOUT frailty the RTTE likelihood therefore reduces
# EXACTLY to an ordinary Weibull survival regression on the gap durations — so
# `survreg(Surv(gap, event) ~ 1, dist="weibull")` is the closed-form MLE that a ferx
# fixed-effects clock-reset fit must reproduce (the gap-time analogue of the standalone
# tte_weibull survreg anchor).
#
# survreg parameterises  log T = intercept + scale_sr * W  (W ~ standard extreme value).
# Map to ferx's H(t) = (t/scale)^shape:  shape = 1/scale_sr,  scale = exp(intercept).
#
# Run:  Rscript survreg.R

suppressMessages(library(survival))
dat <- read.csv("rtte_weibull_reset.csv")
dat <- dat[order(dat$ID, dat$TIME), ]

# Inter-event gaps within each subject (first gap measured from t = 0).
gap   <- ave(dat$TIME, dat$ID, FUN = function(x) c(x[1], diff(x)))
event <- dat$DV   # 1 = observed gap ending in an event, 0 = final censored gap

fit <- survreg(Surv(gap, event) ~ 1, dist = "weibull")

ferx_shape <- 1 / fit$scale
ferx_scale <- exp(unname(coef(fit)[1]))
m2ll       <- -2 * as.numeric(logLik(fit))

cat(sprintf("gaps: %d total (%d events, %d censored)\n",
            length(gap), sum(event == 1), sum(event == 0)))
cat(sprintf("[fixed-effects clock-reset Weibull MLE]  scale = %.5f  shape = %.5f  -2logL = %.4f\n",
            ferx_scale, ferx_shape, m2ll))
cat(sprintf("data-generating: scale = 5.0, shape = 1.5 (mixed, so the pooled fixed fit is biased)\n"))
