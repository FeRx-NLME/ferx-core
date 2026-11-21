# Reference for tests/cif_survfit_reference.rs (#1887): R survival::survfit
# multi-state (Aalen-Johansen) rows for two competing causes on seeded synthetic
# data with two strata. Event times are rounded up and censoring times drawn on
# whole units, so ties (events with events, events with censorings) are frequent.
# Run from this directory:
#   Rscript make_reference.R
# Generated with R 4.6.1, survival 3.8.12.
suppressMessages(library(survival))
set.seed(1887)
n <- 160
stratum <- rep(c("A", "B"), each = n / 2)
rate1 <- ifelse(stratum == "A", 0.10, 0.06)
rate2 <- ifelse(stratum == "A", 0.06, 0.10)
t1 <- rexp(n, rate1)
t2 <- rexp(n, rate2)
cz <- sample(4:20, n, replace = TRUE)
te <- ceiling(pmin(t1, t2))
time <- pmin(te, cz)
cause <- ifelse(te > cz, 0L, ifelse(t1 < t2, 1L, 2L))
data <- data.frame(id = seq_len(n), stratum = stratum, time = time, cause = cause)
write.csv(data, "data.csv", row.names = FALSE, quote = FALSE)

g <- function(x) sprintf("%.17g", x)
rows <- do.call(rbind, lapply(c("A", "B"), function(s) {
  d <- data[data$stratum == s, ]
  f <- survfit(Surv(time, factor(cause, 0:2)) ~ 1, data = d)
  # n.censor is per state in a multi-state fit; censorings leave from state (s0).
  data.frame(stratum = s, time = f$time, n_risk = f$n.risk[, 1],
             n_event1 = f$n.event[, 2], n_event2 = f$n.event[, 3],
             n_censor = f$n.censor[, 1], surv = g(f$pstate[, 1]),
             cif1 = g(f$pstate[, 2]), cif2 = g(f$pstate[, 3]))
}))
# The multi-state fit's all-cause column must equal the ordinary Kaplan-Meier fit.
for (s in c("A", "B")) {
  d <- data[data$stratum == s, ]
  km <- survfit(Surv(time, cause > 0) ~ 1, data = d)
  stopifnot(max(abs(km$surv - as.numeric(rows$surv[rows$stratum == s]))) < 1e-15)
}
write.csv(rows, "reference.csv", row.names = FALSE, quote = FALSE)
mixed <- sum((rows$n_event1 + rows$n_event2) > 0 & rows$n_censor > 0)
cat(sprintf("%d subjects, %d reference rows, %d rows with >1 event, %d event+censor rows\n",
            n, nrow(rows), sum(rows$n_event1 + rows$n_event2 > 1), mixed))
