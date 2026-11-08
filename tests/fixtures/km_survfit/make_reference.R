# Reference for tests/km_survfit_reference.rs (#1886): R survival::survfit
# Kaplan-Meier and Nelson-Aalen rows on seeded synthetic data with two strata.
# Times are rounded to whole units so ties (events with events, events with
# censorings) are frequent. Run from this directory:
#   Rscript make_reference.R
# Generated with R 4.6.1, survival 3.8.12.
suppressMessages(library(survival))
set.seed(1886)
n <- 120
stratum <- rep(c("A", "B"), each = n / 2)
rate <- ifelse(stratum == "A", 0.10, 0.20)
t_event <- rexp(n, rate)
t_cens <- runif(n, 2, 25)
time <- round(pmin(t_event, t_cens))
event <- as.integer(t_event <= t_cens)
data <- data.frame(id = seq_len(n), stratum = stratum, time = time, event = event)
write.csv(data, "data.csv", row.names = FALSE, quote = FALSE)

rows <- do.call(rbind, lapply(c("A", "B"), function(s) {
  d <- data[data$stratum == s, ]
  f <- survfit(Surv(time, event) ~ 1, data = d)
  data.frame(stratum = s, time = f$time, n_risk = f$n.risk, n_event = f$n.event,
             n_censor = f$n.censor, surv = sprintf("%.17g", f$surv),
             cumhaz = sprintf("%.17g", f$cumhaz))
}))
# The stratified fit must reproduce the per-stratum fits row for row.
fs <- survfit(Surv(time, event) ~ stratum, data = data)
# (R's own string-to-double is not always correctly rounded, so this reads back within 1e-15.)
stopifnot(max(abs(fs$surv - as.numeric(rows$surv))) < 1e-15)
write.csv(rows, "reference.csv", row.names = FALSE, quote = FALSE)
cat(sprintf("%d subjects, %d reference rows, %d tied event rows\n",
            n, nrow(rows), sum(rows$n_event > 1)))
