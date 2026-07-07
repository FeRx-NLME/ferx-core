# count_equivalence.R — exact, license-free anchor for constant-hazard RTTE.
#
# For a constant per-subject hazard lambda_i over [0, T], the clock-forward RTTE
# likelihood factorises: the event COUNT N_i ~ Poisson(lambda_i * T), and the event
# times given N_i are uniform on [0, T] (independent of lambda). So the (lambda,
# omega^2) maximiser of the full RTTE likelihood equals the maximiser of the
# per-subject counts under a Poisson-lognormal model. That gives an exact cross-tool
# anchor for BOTH the pooled rate and the frailty variance without any RTTE-specific
# software (only base R + lme4).
#
# Run:  Rscript count_equivalence.R

suppressMessages(library(lme4))
dat <- read.csv("rtte_exp.csv")
horizon <- 20.0

counts <- aggregate(DV ~ ID, data = dat, FUN = function(x) sum(x == 1))
names(counts)[2] <- "n"
counts$logT <- log(horizon)

cat(sprintf("subjects=%d  total events=%d  mean events/subj=%.3f\n",
            nrow(counts), sum(counts$n), mean(counts$n)))

# --- Fixed effects: pooled Poisson rate (no frailty) ---
m0 <- glm(n ~ 1, offset = logT, family = poisson, data = counts)
lam_fe <- exp(coef(m0)[["(Intercept)"]])
cat(sprintf("\n[Fixed effects]  Poisson GLM   lambda_hat = %.5f   (analytic D/exposure = %.5f)\n",
            lam_fe, sum(counts$n) / (nrow(counts) * horizon)))

# --- Mixed: Poisson-lognormal frailty GLMM ---
m1 <- glmer(n ~ 1 + (1 | ID), offset = logT, family = poisson, data = counts)
lam_me <- exp(lme4::fixef(m1)[["(Intercept)"]])
omega2 <- as.numeric(lme4::VarCorr(m1)$ID)
cat(sprintf("[Mixed]          Poisson-LN GLMM   TVLAMBDA = %.5f   omega^2(log rate) = %.5f\n",
            lam_me, omega2))
cat(sprintf("                 data-generating   TVLAMBDA = 0.15000   omega^2           = 0.09000\n"))
