# Prototype of tests/fixtures/kmmc_survfit (#1886 PR3): survfit risk sets, KMMC
# mean over each risk set computed in R. Covariates are in hundredths so a
# Python fractions check can be exact.
suppressMessages(library(survival))
set.seed(18863)
n <- 120
stratum <- rep(c("A", "B"), each = n / 2)
wt <- round(rnorm(n, 70, 12), 2); sex <- rbinom(n, 1, 0.5)
rate <- ifelse(stratum == "A", 0.10, 0.20) * exp(0.03 * (wt - 70) + 0.5 * sex)
t_event <- rexp(n, rate); t_cens <- runif(n, 2, 25)
time <- round(pmin(t_event, t_cens)); event <- as.integer(t_event <= t_cens)
data <- data.frame(id = seq_len(n), stratum, time, event, wt, sex)
write.csv(data, "data.csv", row.names = FALSE, quote = FALSE)
rows <- do.call(rbind, lapply(c("A", "B"), function(s) {
  d <- data[data$stratum == s, ]; f <- survfit(Surv(time, event) ~ 1, data = d)
  data.frame(stratum = s, time = f$time, n_risk = f$n.risk, n_event = f$n.event, n_censor = f$n.censor,
    mean_wt = sprintf("%.17g", sapply(f$time, function(t) mean(d$wt[d$time >= t]))),
    mean_sex = sprintf("%.17g", sapply(f$time, function(t) mean(d$sex[d$time >= t]))))
}))
# survfit's n.risk is the set the mean is taken over:
chk <- mapply(function(s, t) sum(data$stratum == s & data$time >= t), rows$stratum, rows$time)
stopifnot(all(chk == rows$n_risk))
write.csv(rows, "reference.csv", row.names = FALSE, quote = FALSE)
cat(sprintf("%d rows, %d tied-event rows, %d event+censor rows, censored %.0f%%\n", nrow(rows),
  sum(rows$n_event > 1), sum(rows$n_event > 0 & rows$n_censor > 0), 100 * mean(event == 0)))
