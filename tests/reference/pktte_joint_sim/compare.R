#!/usr/bin/env Rscript
# Slice 2.2 event-time SIMULATION anchor (#564): ferx vs NONMEM $SIM.
# Both simulate the identical joint PK-TTE design (each draws its own RNG), so we compare
# the event-time DISTRIBUTIONS. NONMEM provides the per-subject cumulative hazard CHZ(t)
# (sim.tab); its analytic marginal survival S(t) = mean_i exp(-CHZ_i(t)) is RNG-free, so
# we anchor on max |ferx KM(t) - NONMEM S(t)| (seed-robust) — NOT a two-sample KS p-value,
# which would wobble around 0.05 on reseed for no real reason. The sampler's exactness is
# proven separately by the PIT/KS unit test; this is external cross-tool corroboration.
suppressMessages(library(survival))
HORIZON <- 24

ferx <- read.csv("ferx_events.csv")                      # ID,TIME,DV from pktte_sim_anchor
nm   <- read.table("sim.tab", skip = 1, header = TRUE)   # ID TIME CMT CHZ
nm   <- nm[nm$CMT == 3, ]

# NONMEM analytic marginal survival on the grid (no inverse-CDF / no RNG).
ag <- aggregate(CHZ ~ TIME, data = nm, FUN = function(c) mean(exp(-c)))
names(ag)[2] <- "S_nm"; ag <- ag[order(ag$TIME), ]
S_ferx <- summary(survfit(Surv(TIME, DV) ~ 1, data = ferx), times = ag$TIME, extend = TRUE)$surv
dS <- abs(S_ferx - ag$S_nm)

cat(sprintf("ferx   : N=%d  events=%d (%.1f%%)\n", nrow(ferx), sum(ferx$DV), 100*mean(ferx$DV)))
cat(sprintf("NONMEM : N=%d  P(event by 24)=1-S(24)=%.3f (analytic)\n",
            length(unique(nm$ID)), 1 - tail(ag$S_nm, 1)))
cat(sprintf("\nmax |ferx KM(t) - NONMEM S(t)| = %.4f  (at t=%.2f);  mean = %.4f over %d grid pts\n",
            max(dS), ag$TIME[which.max(dS)], mean(dS), nrow(ag)))
g <- c(2, 6, 12, 18, 24); idx <- match(g, ag$TIME)
tab <- rbind(`ferx KM` = S_ferx[idx], `NONMEM S(t)` = ag$S_nm[idx], `|diff|` = dS[idx])
colnames(tab) <- paste0("S(", g, ")"); cat("\n"); print(round(tab, 4))
# Descriptive only (NOT a gate): two-sample KS on the event times.
k <- suppressWarnings(ks.test(ferx$TIME[ferx$DV == 1],
                              { set.seed(424242); n <- length(unique(nm$ID)); tg <- -log(runif(n))
                                ev <- sapply(sort(unique(nm$ID)), function(id){ s<-nm[nm$ID==id,]; s<-s[order(s$TIME),]
                                  if (max(s$CHZ) < tg[which(sort(unique(nm$ID))==id)]) NA else
                                  approx(s$CHZ, s$TIME, tg[which(sort(unique(nm$ID))==id)], ties=min)$y })
                                ev[!is.na(ev)] }))
cat(sprintf("\n[descriptive] two-sample KS on event times: D=%.4f p=%.3f\n", k$statistic, k$p.value))
