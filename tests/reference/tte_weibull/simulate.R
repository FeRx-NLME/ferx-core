# Simulate a standalone Weibull TTE dataset for ferx Phase 1 validation.
#
# True parameters (scale parameterization: H(t) = (t/scale)^shape):
#   log_scale_pop = log(20)   (scale_pop = 20 time-units)
#   log_shape_pop = log(2.0)  (shape_pop = 2.0; increasing hazard)
#   omega_shape   = 0.20      (variance of log(shape_i); shape is subject-varying)
#
# Individual parameter: shape_i = 2.0 * exp(eta_i),  eta_i ~ N(0, 0.20)
# Scale is fixed (no ETA on scale for Phase 1 validation simplicity).
# Right-censoring at t = 30.
#
# Weibull inverse-CDF: T = scale * (-log U)^(1/shape_i)
#
# Output: tte_weibull.csv (NONMEM-compatible, one row per subject)
#
# Run:  Rscript simulate.R

set.seed(42)

n         <- 100
scale_pop <- 20.0
shape_pop <- 2.0
omega_shape <- 0.20
t_censor  <- 30

eta_shape <- rnorm(n, mean = 0, sd = sqrt(omega_shape))
shape_i   <- shape_pop * exp(eta_shape)
# scale is fixed (no between-subject variability on scale)

u       <- runif(n)
t_event <- scale_pop * (-log(u))^(1 / shape_i)

t_obs <- pmin(t_event, t_censor)
event <- as.integer(t_event <= t_censor)

df <- data.frame(
  ID   = seq_len(n),
  TIME = round(t_obs, 4),
  DV   = event,
  EVID = 0,
  CMT  = 2,
  MDV  = 0
)

cat(sprintf("N = %d, events = %d (%.0f%%), censored = %d\n",
            n, sum(event), 100 * mean(event), sum(1 - event)))
# Population median survival at mean shape: t = scale * log(2)^(1/shape_pop) = 20 * 0.5^0.5 ≈ 14.1
cat(sprintf("Approx population median: %.2f\n", scale_pop * log(2)^(1/shape_pop)))

write.csv(df, "tte_weibull.csv", row.names = FALSE, quote = FALSE)
cat("Wrote tte_weibull.csv\n")
