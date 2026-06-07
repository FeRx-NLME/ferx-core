# Simulate a standalone Exponential TTE dataset for ferx Phase 1 validation.
#
# True parameters:
#   log_lambda_pop = log(0.1)   (lambda_pop = 0.1 event/time-unit)
#   omega_lambda   = 0.25       (variance of log(lambda_i))
#
# Individual parameter: lambda_i = 0.1 * exp(eta_i),  eta_i ~ N(0, 0.25)
# Right-censoring at t = 24 (administrative).
# Expected ~30% censoring (P(T > 24) ≈ exp(-0.1 * 24) ≈ 0.09 at population mean,
# higher after integrating over eta).
#
# Output: tte_exp.csv  (NONMEM-compatible, one row per subject)
#
# Run:  Rscript simulate.R
#       Output committed as tte_exp.csv

set.seed(42)

n        <- 100
lambda_pop <- 0.1
omega_lambda <- 0.25          # variance on log scale
t_censor    <- 24

eta   <- rnorm(n, mean = 0, sd = sqrt(omega_lambda))
lam_i <- lambda_pop * exp(eta)

# Exact event time from Exponential CDF inverse: T = -log(U) / lambda_i
u       <- runif(n)
t_event <- -log(u) / lam_i

t_obs  <- pmin(t_event, t_censor)
event  <- as.integer(t_event <= t_censor)   # 1 = exact event, 0 = right-censored

df <- data.frame(
  ID   = seq_len(n),
  TIME = round(t_obs, 4),
  DV   = event,       # 0 = censored, 1 = exact event
  EVID = 0,
  CMT  = 2,           # CMT=2 → [event_model] cmt = 2 in .ferx file
  MDV  = 0
)

cat(sprintf("N = %d, events = %d (%.0f%%), censored = %d\n",
            n, sum(event), 100 * mean(event), sum(1 - event)))
cat(sprintf("Empirical median survival: %.2f (expected: %.2f)\n",
            median(t_obs[event == 1]), log(2) / lambda_pop))

write.csv(df, "tte_exp.csv", row.names = FALSE, quote = FALSE)
cat("Wrote tte_exp.csv\n")
