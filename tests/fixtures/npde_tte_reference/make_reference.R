# Reference for the single-event TTE npde kernel (ferx-core #1889, test T5 in
# src/stats/npde_tte_tests.rs).
#
# The R `npde` package (CRAN 3.5) has no time-to-event mode. It is used here through
# the negation trick: right censoring of T at c is left censoring of y' = -T at
# LOQ = -c, which `autonpde(cens.method = "cdf")` handles. The package applies ONE
# LOQ (the smallest censored value) to every subject, so the trick is exact only
# when every censored subject shares one censoring time. This fixture therefore
# uses administrative censoring at a single time C.
#
# Run from this directory:  Rscript make_reference.R
# Writes obs.csv, sims.csv and reference.csv. Seeded, so a re-run reproduces them.
suppressMessages(library(npde))
set.seed(1889)
lam0 <- 0.1; om2 <- 0.25; K <- 200; N <- 40; C <- 6
sim_T <- function(n) rexp(n, lam0 * exp(rnorm(n, 0, sqrt(om2))))

tt <- sim_T(N)
sims <- matrix(sim_T(N * K), nrow = N)            # sims[i, k], unbounded draws

# Two edge rows, so both clamp ends are exercised: subject 1 has an event before
# every one of its simulated times, subject 2 an event after all of them.
tt[1] <- 1e-6
tt[2] <- 5
sims[2, ] <- runif(K, 0.01, 4.9)

y  <- pmin(tt, C)
ev <- tt <= C
stopifnot(ev[1], ev[2], sum(!ev) >= 5, all(sims[1, ] > y[1]), all(sims[2, ] < y[2]))

obs  <- data.frame(id = 1:N, x = 1, y = -y, cens = as.integer(!ev))
simd <- data.frame(idsim = rep(1:N, K), xsim = 1, ysim = as.vector(-sims))
r <- autonpde(obs, simd, iid = 1, ix = 2, iy = 3, icens = 4, boolsave = FALSE,
              cens.method = "cdf", calc.npde = TRUE, calc.npd = TRUE, verbose = FALSE)
res <- r@results@res
stopifnot(nrow(res) == N)

g <- function(v) sprintf("%.17g", v)
write.csv(data.frame(id = 1:N, time = g(y), event = as.integer(ev)),
          "obs.csv", row.names = FALSE, quote = FALSE)
# What the engine's simulator hands the kernel: a draw at or past the horizon (the
# latest observed time, C here) is reported AT the horizon with no event.
H <- max(y)
long <- data.frame(id = rep(1:N, K), rep = rep(1:K, each = N),
                   time = g(pmin(as.vector(sims), H)),
                   observed = as.integer(as.vector(sims) < H))
write.csv(long[order(long$id, long$rep), ], "sims.csv", row.names = FALSE, quote = FALSE)
write.csv(data.frame(id = 1:N, pd_R = g(res$pd), ploq_R = g(r@results@ploq),
                     npde_R = g(res$npde), npd_R = g(res$npd)),
          "reference.csv", row.names = FALSE, quote = FALSE)
cat("npde", as.character(packageVersion("npde")), R.version.string, "\n")
cat("events", sum(ev), "censored", sum(!ev), "loq", r@data@loq, "horizon", H, "\n")
