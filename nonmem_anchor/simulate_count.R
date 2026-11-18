# Simulates the #2877 count-endpoint fixtures (count_pois.csv, count_nb.csv).
# Run from nonmem_anchor/:  Rscript simulate_count.R
# Design: 60 subjects x 6 records; WT varies between subjects, TRT within a subject
# (0 on the first three records, 1 on the last three) so the mean reads a covariate that
# changes across records as well as across subjects.
#   log(lambda) = log(BASE) + BETA*(WT/70 - 1) + GAM*TRT + ETA,  ETA ~ N(0, OMEGA)
set.seed(2877)
n <- 60; times <- c(0, 7, 14, 21, 28, 35)
wt <- round(rlnorm(n, log(70), 0.25), 1)
eta <- rnorm(n, 0, sqrt(0.25))
mk <- function(draw) {
  rows <- list()
  for (i in seq_len(n)) {
    trt <- c(0, 0, 0, 1, 1, 1)
    lam <- exp(log(4) + 0.8 * (wt[i] / 70 - 1) - 0.5 * trt + eta[i])
    rows[[i]] <- data.frame(ID = i, TIME = times, DV = draw(lam), CMT = 1, EVID = 0, MDV = 0,
                            WT = wt[i], TRT = trt)
  }
  do.call(rbind, rows)
}
pois <- mk(function(l) rpois(length(l), l))
nb <- mk(function(l) rnbinom(length(l), size = 3, mu = l))
write.csv(pois, "count_pois.csv", row.names = FALSE, quote = FALSE)
write.csv(nb, "count_nb.csv", row.names = FALSE, quote = FALSE)
