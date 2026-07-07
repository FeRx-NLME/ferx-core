# simulate.R — constant-hazard repeated time-to-event (RTTE) with subject frailty.
#
# Phase 3 Slice 3.1 clock-forward RTTE anchor dataset.
#
# Model: subject i has frailty eta_i ~ N(0, omega2) and an individual constant
# hazard lambda_i = TVLAMBDA * exp(eta_i). Events recur with i.i.d. Exponential
# inter-event gaps; every subject is observed over [0, horizon] and is at risk the
# whole window (recurrent events do not remove the subject). Each realised event is a
# DV=1 row; a final DV=0 row administratively right-censors at `horizon`.
#
# Clock-forward (Andersen-Gill): the hazard is a function of absolute time (here a
# constant), and the cumulative hazard is NOT reset at events.

set.seed(31)

N        <- 100L    # subjects
TVLAMBDA <- 0.15    # population event rate (1/time)
omega2   <- 0.09    # frailty variance (variance of log-rate)
horizon  <- 20.0    # administrative right-censoring time

rows <- list()
k <- 0L
for (i in seq_len(N)) {
  eta      <- rnorm(1, 0, sqrt(omega2))
  lambda_i <- TVLAMBDA * exp(eta)
  t <- 0
  repeat {
    t <- t + rexp(1, rate = lambda_i)
    if (t >= horizon) break
    k <- k + 1L
    rows[[k]] <- data.frame(ID = i, TIME = round(t, 4), DV = 1L, EVID = 0L, CMT = 2L, MDV = 0L)
  }
  # Administrative right-censor at the horizon (DV=0).
  k <- k + 1L
  rows[[k]] <- data.frame(ID = i, TIME = horizon, DV = 0L, EVID = 0L, CMT = 2L, MDV = 0L)
}

dat <- do.call(rbind, rows)
write.csv(dat, "rtte_exp.csv", row.names = FALSE, quote = FALSE)

n_event <- sum(dat$DV == 1L)
exposure <- N * horizon                       # each subject at risk over [0, horizon]
lambda_hat <- n_event / exposure              # pooled constant-hazard MLE (fixed effects)

cat(sprintf("subjects=%d  events=%d  mean events/subj=%.3f\n", N, n_event, n_event / N))
cat(sprintf("exposure=%.1f  fixed-effects MLE lambda_hat = D/exposure = %.5f\n",
            exposure, lambda_hat))
cat(sprintf("mean lambda_i (mixed) = TVLAMBDA*exp(omega2/2) = %.5f\n",
            TVLAMBDA * exp(omega2 / 2)))
