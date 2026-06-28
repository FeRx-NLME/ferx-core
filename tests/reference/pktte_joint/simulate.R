#!/usr/bin/env Rscript
# Simulate a known-truth joint PK-TTE dataset for the ferx anchor (#564).
#
# Oral 1-cpt PK + drug-driven hazard  h(t) = H0 * exp(BETA * Cc),  Cc = central/V.
# Cc uses the closed-form 1-cpt oral solution (no ODE solver / C compiler needed, so
# this reproduces anywhere); the cumulative hazard is the trapezoidal integral of h on
# a fine grid, and event times are drawn by inverting H(T) = -log(U). PK is observed on
# CMT 2 (concentration); the single (possibly right-censored) event is on CMT 3.
#
# Run:  Rscript simulate.R   ->   pktte_joint.csv  (NONMEM-format, shared by all tools)
set.seed(20260628)

## ---- truth ----
TVCL <- 1.0; TVV <- 10.0; TVKA <- 1.0
H0   <- 0.015; BETA <- 0.25
om_CL <- 0.09; prop_sd <- 0.10
N <- 120; dose <- 100; horizon <- 24
obs_t <- c(0.5, 2, 6, 12)

# Closed-form 1-cpt oral concentration (F = 1).
cc_fun <- function(t, CL, V, KA) {
  ke <- CL / V
  (dose * KA) / (V * (KA - ke)) * (exp(-ke * t) - exp(-KA * t))
}

fine <- seq(0, horizon, by = 0.01)
out  <- vector("list", N)
nev  <- 0L
for (i in 1:N) {
  CLi <- TVCL * exp(rnorm(1, 0, sqrt(om_CL)))
  cc  <- cc_fun(fine, CLi, TVV, TVKA)
  h   <- H0 * exp(BETA * cc)
  H   <- c(0, cumsum((h[-1] + h[-length(h)]) / 2 * diff(fine)))  # cumulative hazard
  u   <- runif(1); tgt <- -log(u)
  k   <- which(H >= tgt)[1]
  if (is.na(k)) { Te <- horizon; dv <- 0L
  } else {
    Te <- approx(H[(k - 1):k], fine[(k - 1):k], tgt)$y; dv <- 1L
  }
  if (!is.finite(Te) || Te > horizon) { Te <- horizon; dv <- 0L }
  nev <- nev + dv
  cpo <- cc_fun(obs_t, CLi, TVV, TVKA) * (1 + rnorm(length(obs_t), 0, prop_sd))
  rid <- rbind(
    data.frame(ID = i, TIME = 0,     DV = NA, EVID = 1, AMT = dose, CMT = 1, MDV = 1),
    data.frame(ID = i, TIME = obs_t, DV = cpo, EVID = 0, AMT = 0,    CMT = 2, MDV = 0),
    data.frame(ID = i, TIME = Te,    DV = dv, EVID = 0, AMT = 0,    CMT = 3, MDV = 0)
  )
  out[[i]] <- rid[order(rid$TIME), ]
}
dat <- do.call(rbind, out)
dat$DV <- ifelse(is.na(dat$DV), ".", trimws(formatC(dat$DV, format = "g", digits = 6)))
write.csv(dat, "pktte_joint.csv", row.names = FALSE, quote = FALSE)
cat(sprintf("truth: CL=%.3g V=%.3g KA=%.3g H0=%.3g BETA=%.3g om_CL=%.3g prop_sd=%.3g\n",
            TVCL, TVV, TVKA, H0, BETA, om_CL, prop_sd))
cat(sprintf("N=%d  events=%d (%.0f%%)  censored=%d  rows=%d\n",
            N, nev, 100 * nev / N, N - nev, nrow(dat)))
