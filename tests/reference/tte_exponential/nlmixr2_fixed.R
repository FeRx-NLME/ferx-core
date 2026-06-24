# FIXED-EFFECTS Exponential TTE fit with nlmixr2 (no random effect).
# Matches the NONMEM run that actually completed ($OMEGA 0 FIX, METHOD=0) and the
# ferx n_eta=0 / survreg anchor. Keep the frailty fit (nlmixr2.R) as the separate
# mixed-effects comparison.
#
# Run:  Rscript nlmixr2_fixed.R

library(nlmixr2)

dat <- read.csv("tte_exp.csv")   # ID,TIME,DV,EVID,CMT,MDV ; DV = event indicator
# nlmixr2 needs the standard DV/TIME columns present; the ll(tte) model references an
# `event` column, so mirror DV into it (keep DV/TIME so the data check passes).
dat$event <- dat$DV

expo_fixed <- function() {
  ini({
    log_lambda <- log(0.1)
  })
  model({
    lambda <- exp(log_lambda)
    h      <- lambda
    H      <- lambda * time
    ll(tte) ~ event * log(h) - H
  })
}

fit <- nlmixr(expo_fixed, dat, est = "focei", control = foceiControl(print = 0))

print(fit)
npar <- length(fixef(fit))
ll   <- fixef(fit)["log_lambda"]
cat("\n--- FIXED-EFFECTS Exponential (nlmixr2 focei) ---\n")
cat(sprintf("log_lambda:        %.5f\n", ll))
cat(sprintf("lambda (rate):     %.6f\n", exp(ll)))
cat(sprintf("raw OBJF:          %.4f\n", fit$objective))
cat(sprintf("AIC:               %.4f\n", fit$AIC))
cat(sprintf("-2LL (AIC-2*npar): %.4f   [npar=%d]\n", fit$AIC - 2 * npar, npar))
