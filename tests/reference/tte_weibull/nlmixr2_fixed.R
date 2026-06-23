# FIXED-EFFECTS Weibull TTE fit with nlmixr2 (no random effect on shape).
# Matches the NONMEM run that actually completed ($OMEGA 0 FIX, METHOD=0) and the
# ferx n_eta=0 / survreg anchor. Keep the frailty fit (nlmixr2.R) as the separate
# mixed-effects comparison.
#
# Scale parameterization: H(t) = (t/scale)^shape, h(t) = (shape/scale)*(t/scale)^(shape-1)
# Run:  Rscript nlmixr2_fixed.R

library(nlmixr2)

dat <- read.csv("tte_weibull.csv")  # ID,TIME,DV,EVID,CMT,MDV ; DV = event indicator
# nlmixr2 needs the standard DV/TIME columns present; the ll(tte) model references an
# `event` column, so mirror DV into it (keep DV/TIME so the data check passes).
dat$event <- dat$DV

weibull_fixed <- function() {
  ini({
    log_scale <- log(20)
    log_shape <- log(2)
  })
  model({
    scale <- exp(log_scale)
    shape <- exp(log_shape)
    h <- (shape / scale) * (time / scale)^(shape - 1)
    H <- (time / scale)^shape
    ll(tte) ~ event * log(h) - H
  })
}

fit <- nlmixr(weibull_fixed, dat, est = "focei", control = foceiControl(print = 0))

print(fit)
npar <- length(fixef(fit))
ls_  <- fixef(fit)["log_scale"]
lsh  <- fixef(fit)["log_shape"]
cat("\n--- FIXED-EFFECTS Weibull (nlmixr2 focei) ---\n")
cat(sprintf("log_scale: %.5f   scale: %.4f\n", ls_, exp(ls_)))
cat(sprintf("log_shape: %.5f   shape: %.4f\n", lsh, exp(lsh)))
cat(sprintf("raw OBJF:          %.4f\n", fit$objective))
cat(sprintf("AIC:               %.4f\n", fit$AIC))
cat(sprintf("-2LL (AIC-2*npar): %.4f   [npar=%d]\n", fit$AIC - 2 * npar, npar))
