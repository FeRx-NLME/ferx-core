# Simulate a standalone Gompertz TTE dataset for ferx Phase 1 validation.
#
# True parameters (matches structure of nlmixr2 blog reference, §7.6 of plan):
#   log_alpha_pop = -6.0   (alpha_pop ≈ 0.00248; baseline hazard at t=0)
#   log_gamma_pop = -5.4   (gamma_pop ≈ 0.00450; hazard growth rate)
#   log_hr        = -0.8   (HR ≈ 0.449; 2-arm RCT treatment effect)
#
# h(t) = alpha * exp(gamma * t) * exp(log_hr * trt)
# H(t) = (alpha/gamma) * (exp(gamma*t) - 1) * exp(log_hr * trt)
#
# No random effects (fixed-effects model to match nlmixr2 blog reference).
# Administrative censoring at t = 365 days.
# n = 300 subjects, 2-arm RCT (trt = 0 or 1, balanced).
#
# Gompertz inverse-CDF: T = (1/gamma) * log(1 - (gamma/alpha) * log(U) / exp(log_hr*trt))
# (derived from H(T) = -log U)
#
# Output: tte_gompertz.csv
#
# Cross-check: the nlmixr2 blog (blog.nlmixr2.org/blog/2026-05-28-survival-nlmixr2/)
# uses a Gompertz model with similar structure and reports OFV = 2955.64 for 300 subjects.
# Estimate differences between that blog's dataset and this one are expected (different seeds
# / true params), but the scale and estimator behaviour should be comparable.
#
# Run:  Rscript simulate.R

set.seed(42)

n       <- 300
alpha   <- exp(-6.0)   # ≈ 0.00248
gamma   <- exp(-5.4)   # ≈ 0.00450
log_hr  <- -0.8        # HR ≈ 0.449 for treated vs control
t_censor <- 365

trt     <- rep(c(0, 1), n / 2)   # balanced 2-arm

# Exact Gompertz inverse-CDF under exponential(log_hr*trt) proportional hazard:
# H(T) = (alpha / gamma) * (exp(gamma*T) - 1) * exp(log_hr * trt) = -log(U)
# => exp(gamma*T) = 1 + (gamma * (-log U)) / (alpha * exp(log_hr * trt))
# => T = (1/gamma) * log(1 + (gamma * (-log U)) / (alpha * exp(log_hr * trt)))
u         <- runif(n)
hazard_adj <- alpha * exp(log_hr * trt)
inner     <- 1 + (gamma * (-log(u))) / hazard_adj
t_event   <- (1 / gamma) * log(pmax(inner, 1 + .Machine$double.eps))

t_obs <- pmin(t_event, t_censor)
event <- as.integer(t_event <= t_censor)

df <- data.frame(
  ID   = seq_len(n),
  TIME = round(t_obs, 2),
  DV   = event,
  TRT  = trt,
  EVID = 0,
  CMT  = 2,
  MDV  = 0
)

cat(sprintf("N = %d, events = %d (%.0f%%), censored = %d\n",
            n, sum(event), 100 * mean(event), n - sum(event)))
cat(sprintf("Events by arm: control=%d, treated=%d\n",
            sum(event[trt == 0]), sum(event[trt == 1])))

write.csv(df, "tte_gompertz.csv", row.names = FALSE, quote = FALSE)
cat("Wrote tte_gompertz.csv\n")
