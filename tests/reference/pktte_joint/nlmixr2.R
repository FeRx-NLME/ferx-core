# Fit the joint PK-TTE dataset with nlmixr2 — reference for ferx Slice 2.1 (#564).
#
# Same model as examples/pktte_joint.ferx: oral 1-cpt PK (concentration on CMT 2) +
# a drug-driven hazard h = H0*exp(BETA*Cc) accumulated as an ODE state, with the event
# on CMT 3. FOCEI, joint Gaussian-PK + TTE likelihood, shared eta on CL.
#
# Requires: nlmixr2, rxode2, and a working C/Fortran toolchain. On a machine where R's
# configured gfortran is missing, point FLIBS at the installed one, e.g.:
#   R_MAKEVARS_USER=/path/to/Makevars Rscript nlmixr2.R
# with Makevars containing: FLIBS=-L/usr/local/gfortran/lib -lgfortran -lquadmath
#
# Run:  Rscript nlmixr2.R   (paste key output into expected.md)

library(nlmixr2)

dat <- read.csv("pktte_joint.csv", na.strings = ".")
# The ll(tte) endpoint references an `event` indicator; mirror DV (1=event,0=cens on the
# CMT-3 rows). On PK rows `event` is unused (the tte endpoint is routed to CMT 3 only).
dat$event <- dat$DV
# nlmixr2 numbers ODE compartments depot=1, central=2, cumhaz=3, so its `tte` endpoint
# lands on compartment 4 — whereas the shared (ferx-convention) dataset labels the event
# rows CMT 3. Remap only the event rows so nlmixr2 routes them to its tte endpoint; PK rows
# stay on CMT 2 (= central) and the data values are identical across tools.
dat$CMT[dat$CMT == 3] <- 4

joint_model <- function() {
  ini({
    lcl  <- log(1.0)
    lv   <- log(10.0)
    lka  <- log(1.0)
    lh0  <- log(0.015)
    beta <- 0.25
    eta.cl ~ 0.09
    prop.sd <- 0.10
  })
  model({
    cl <- exp(lcl + eta.cl)
    v  <- exp(lv)
    ka <- exp(lka)
    h0 <- exp(lh0)
    ke <- cl / v
    d/dt(depot)   = -ka * depot
    d/dt(central) =  ka * depot - ke * central
    d/dt(cumhaz)  =  h0 * exp(beta * central / v)
    cp  = central / v
    haz = h0 * exp(beta * central / v)
    cp ~ prop(prop.sd) | central          # PK observation (CMT 2)
    ll(tte) ~ event * log(haz) - cumhaz   # TTE endpoint (CMT 3)
  })
}

# BOBYQA outer optimizer: FOCEI's gradient-based outer step can "false-converge" on this
# ODE + accumulated-hazard system (nlmixr2's own suggestion); the derivative-free outer
# loop reaches a cleaner optimum for the collinear H0/BETA pair.
fit <- nlmixr(joint_model, dat, est = "focei",
              control = foceiControl(print = 5, outerOpt = "bobyqa"))

print(fit)
cat("\n--- Key values for expected.md ---\n")
cat(sprintf("OFV (-2LL):  %.4f\n", fit$objective))
fe <- fixef(fit)
for (nm in names(fe)) cat(sprintf("%-8s %.5f\n", nm, fe[[nm]]))
cat(sprintf("omega eta.cl (var): %.5f\n", fit$omega["eta.cl", "eta.cl"]))
