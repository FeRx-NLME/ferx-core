# Reference for tests/rtte_survfit_reference.rs (#1886 PR2): R survival::survfit on
# seeded repeated-events data in two strata.
#   - mean cumulative function: survfit(Surv(start, stop, status) ~ 1, id = id)$cumhaz
#   - Kaplan-Meier by event order, absolute time (orders 1..3) and gap time (orders 2..3):
#     survfit(Surv(t, d) ~ 1) on the per-order times
# Times are whole units, so ties across subjects are frequent, and a subject's events
# are drawn without replacement, so no subject has two events at one time (survfit
# refuses stop == start). Some subjects have no event; some have one at their own
# follow-up end. Run from this directory:
#   Rscript make_reference.R
# Generated with R 4.6.1, survival 3.8.12.
suppressMessages(library(survival))
set.seed(18862)
n <- 80
K <- 3
stratum <- rep(c("A", "B"), each = n / 2)
rate <- ifelse(stratum == "A", 0.12, 0.25)
fu <- sample(3:20, n, replace = TRUE)
ev <- lapply(seq_len(n), function(i) {
  k <- min(rpois(1, rate[i] * fu[i]), fu[i])
  sort(sample(seq_len(fu[i]), k))
})

# The data as the model reads them: one row per event, and a closing censoring row
# unless the last event is at the follow-up end.
data <- do.call(rbind, lapply(seq_len(n), function(i) {
  e <- ev[[i]]
  closing <- length(e) == 0 || tail(e, 1) < fu[i]
  data.frame(id = i, stratum = stratum[i],
             time = c(e, if (closing) fu[i]),
             event = c(rep(1L, length(e)), if (closing) 0L))
}))
write.csv(data, "data.csv", row.names = FALSE, quote = FALSE)

fmt <- function(x) sprintf("%.17g", x)
km_rows <- function(s, curve, order, t, d) {
  f <- survfit(Surv(t, d) ~ 1)
  data.frame(stratum = s, curve = curve, order = order, time = f$time, n_risk = f$n.risk,
             n_event = f$n.event, n_censor = f$n.censor, value = fmt(f$surv))
}
rows <- do.call(rbind, lapply(c("A", "B"), function(s) {
  ids <- which(stratum == s)
  cp <- do.call(rbind, lapply(ids, function(i) {
    d <- data[data$id == i, ]
    data.frame(id = i, start = c(0, head(d$time, -1)), stop = d$time, status = d$event)
  }))
  f <- survfit(Surv(start, stop, status) ~ 1, id = id, data = cp)
  out <- data.frame(stratum = s, curve = "mcf", order = 0L, time = f$time, n_risk = f$n.risk,
                    n_event = f$n.event, n_censor = f$n.censor, value = fmt(f$cumhaz))
  for (k in seq_len(K)) {
    t <- sapply(ids, function(i) if (length(ev[[i]]) >= k) ev[[i]][k] else fu[i])
    d <- sapply(ids, function(i) as.integer(length(ev[[i]]) >= k))
    out <- rbind(out, km_rows(s, "abs", k, t, d))
  }
  for (k in 2:K) {
    at <- ids[sapply(ev[ids], length) >= k - 1]
    t <- sapply(at, function(i) {
      (if (length(ev[[i]]) >= k) ev[[i]][k] else fu[i]) - ev[[i]][k - 1]
    })
    d <- sapply(at, function(i) as.integer(length(ev[[i]]) >= k))
    out <- rbind(out, km_rows(s, "gap", k, t, d))
  }
  out
}))
write.csv(rows, "reference.csv", row.names = FALSE, quote = FALSE)

n_ev <- sapply(ev, length)
at_end <- sum(mapply(function(e, c) length(e) > 0 && tail(e, 1) == c, ev, fu))
mcf <- rows[rows$curve == "mcf", ]
stopifnot(sum(n_ev == 0) >= 5, at_end >= 3, sum(mcf$n_event > 1) >= 10,
          all(sapply(ev, function(e) !any(duplicated(e)))))
cat(sprintf("%d subjects, %d data rows, %d reference rows; %d with no event, %d with an event at the follow-up end, %d tied MCF rows; events per subject up to %d\n",
            n, nrow(data), nrow(rows), sum(n_ev == 0), at_end, sum(mcf$n_event > 1), max(n_ev)))
