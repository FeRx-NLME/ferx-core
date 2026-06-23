# License-free fixed-effects (no random-effect) Exponential TTE reference.
#
# Uses base R's recommended `survival` package (no nlmixr2/NONMEM license needed).
# For a censored Exponential the MLE has the closed form  lambda = events / sum(time),
# which is exactly what a ferx fixed-effects (n_eta = 0) Exponential fit maximises.
# This is the tight, apples-to-apples cross-tool anchor for the n_eta=0 path (plan D7);
# it does NOT estimate omega (use the mixed-effects ferx fit + NONMEM/nlmixr2 for that).
#
# Run:  Rscript survreg.R   (from this directory)

library(survival)

dat <- read.csv("tte_exp.csv")          # ID,TIME,DV,EVID,CMT,MDV ; DV: 1=event 0=censored

fit <- survreg(Surv(TIME, DV) ~ 1, data = dat, dist = "exponential")

# survreg uses an AFT log-time parameterisation: log(T) = intercept + error.
# For the exponential, rate lambda = exp(-intercept).
intercept   <- coef(fit)[["(Intercept)"]]
lambda_hat  <- exp(-intercept)
loglambda   <- log(lambda_hat)

# Closed-form check: lambda_MLE = (#events) / (total observed time at risk).
lambda_cf <- sum(dat$DV) / sum(dat$TIME)

# -2 log-likelihood (for OFV cross-reference; constants differ from NONMEM F_FLAG).
m2ll <- -2 * as.numeric(logLik(fit))

cat(sprintf("survreg exponential (fixed effects, no omega):\n"))
cat(sprintf("  intercept        = %.6f\n", intercept))
cat(sprintf("  lambda_hat       = %.6f   (rate)\n", lambda_hat))
cat(sprintf("  log(lambda_hat)  = %.6f\n", loglambda))
cat(sprintf("  lambda closed-fm = %.6f   (events/sum(time)) -- must match lambda_hat\n", lambda_cf))
cat(sprintf("  -2 logLik        = %.6f\n", m2ll))
cat(sprintf("  events / N       = %d / %d\n", sum(dat$DV), nrow(dat)))
