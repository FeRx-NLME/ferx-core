# Modeled-dose × {SS, IOV} and zero-order `dur` × {lagtime, reset, TV-cov} — analytic sensitivities — #486

> **Status:** design **A (modeled-dose + IOV)** is **DONE** on branch `feat/486-modeled-dose-iov`
> — gate `ode_iov_subject_supported` relaxed to the same `has_ss`/`all_slots_present` screen as
> the non-IOV walk; the κ-coupled modeled slot rides the per-occasion stacked jet; FD-diagnostic
> mirror in `inner_optimizer.rs` updated; tests
> `ode_iov_modeled_duration_{provider,kappa_coupled}_matches_fd_of_predict_iov` +
> `ode_iov_modeled_duration_ss_falls_back_to_fd` + two `iov_fd_reason` attribution tests.
> **NONMEM cross-check:** FOCEI modeled-duration + IOV(CL), 40 subj — ferx (analytic, OFV 210.321)
> vs NONMEM `M1 INTER` (OFV 210.298): OFV within 0.011%, all params ≤ 2.4% (see
> `../../nm_modeled_iov/VALIDATION.md`). Designs **B–E** below remain open.

## Goal

The analytic moving infusion-**end** saltation (#630, closes #530) made modeled-duration/rate
doses (`RATE=-1/-2`, `D{cmt}`) and `zero_order(dur)` analytic on the ODE event-driven walk —
but only for base cases. Close the remaining combination gaps (all currently FD):
- **modeled-dose + IOV**
- **modeled-dose + steady-state (SS)**
- **`zero_order`/`mixed` `dur` + lagtime**
- **`zero_order` + reset + input-rate forcing**
- **`zero_order` + time-varying covariates (TV-cov)**

These are five independent FD edges over the **same** #630 saltation primitive; they can ship in
separate PRs. The modeled-dose+IOV one is the cheapest (the IOV walk already routes through the
modeled-dose machinery — only a gate blocks it).

## Current state (what already exists)

All anchors in `src/sens/ode_provider.rs` unless noted.

- **Sign-agnostic saltation kernel** `inject_rate_saltation` (`2802`): `u[cmt] += s·Δr·δ` (1st)
  and `u += -s·½·J·(Δr·e_cmt)·δ²` (2nd, via `jdotg_value` Dual1 directional eval `2828`).
  `s=-1` rate-on, `s=+1` rate-off; `δ` = whatever moves the boundary (`δlag`, `δt_inf`, `δdur`).
  Used by **all four** boundary cases.
- **Moving infusion-end (#630)** `K_INF_END` branch (`3442-3484`) inside `integrate_tvcov_g`
  (`2992`): fires rate-off with `d_off = δlag + δt_inf` (`3468-3469`) over lagtime / rate-defined
  `F` / **modeled** (`is_modeled = modeled_at(idx).is_some()`, gate `3457`).
- **Modeled-dose → live jet** `inf_eff` (`3061-3110`): `ModeledDuration` → window=`D` jet,
  rate=`F·amt/D` (`3079-3083`); `ModeledRate` → rate=`R` jet, window=`F·amt/R` (`3085-3089`).
  Slot wiring `dose_modeled_slot` (`integrate_tvcov_readout:1801-1811`, `modeled_slot_for:448`,
  `modeled_at:3029`).
- **Lagtime moving-START template (#472)**: infusion-start rate-on `s=-1` (`3297-3319`); bolus
  lagtime event-time injection `D·δlag + coef2·δlag²` (`3320-3424`).
- **Static-walk zero-order rate-off (#530)** in `integrate_g` (non-TV twin): `zero_windows`
  build (`3573-3601`, rate=`F·amt·frac/dur`); rate-off at `w_end = w_start + dur` (`3725-3753`);
  window break (`3620-3625`); active-window containment (`3796-3815`).
- **SS dual equilibration (#473)** `equilibrate_ss_state_g` (`2863`, called `3278`): finite
  `SS_EQUILIBRATION_CYCLES` loop; consumes a **fixed per-cycle `t_inf`** (doc `2858-2862`).
  f64 reference `ode/predictions.rs:283`, event-driven `pk/event_driven.rs:339`.
- **Crucial**: the **IOV walk reuses the same** `integrate_tvcov_readout`/`integrate_tvcov_g`
  (`643`, `650`), which already builds `dose_modeled_slot` — so the modeled-dose moving-boundary
  machinery is already wired into the IOV path. Only the support gate stops it.

## Gap (the five FD gates)

- **modeled-dose + IOV** — `ode_iov_subject_supported:1982-1984`: `if !subject.all_doses_fixed()
  { return None; }` (the *only* clause blocking modeled doses under IOV).
- **modeled-dose + SS** — `ode_tvcov_supported:532-540`: the `has_ss` arm of the
  `!all_doses_fixed()` block (`has_ss` at `469`; rationale `525-531`).
- **zero_order/mixed `dur` + lagtime** — `ode_analytical_supported:324-326`:
  `if model.has_lagtime() && !ode.input_rate.is_empty() { return false; }` (static walk assumes
  `w_start = dose.time`, `3571`).
- **zero_order + reset + input-rate** — `ode_subject_supported:386-390`:
  `if !ode.input_rate.is_empty() && !subject.reset_times.is_empty() { return false; }`.
- **zero_order + TV-cov** — two-sided: static walk declines TV-cov (`ode_subject_supported:352`),
  event-driven walk declines any `input_rate` (`ode_tvcov_supported:514-519`) → no analytic walk.

## Design

### A. modeled-dose + IOV (cheapest)
Relax the `all_doses_fixed()` gate at `ode_iov_subject_supported:1982`. The IOV walk already
routes through `integrate_tvcov_g`, so the moving-boundary jet fires. Main new concern: the `D`/`R`
slot can be **κ-coupled** (`D1 = TVD1*exp(ETA_D1 + KAPPA_D1)`); verify `inf_eff` reads the
occasion-seeded stacked jet and the `δdur` column lands in the correct κ-group axis (width bound
`2026-2029`).

### B. modeled-dose + SS
Teach `equilibrate_ss_state_g` (`2863`) to consume a `t_inf = D`-jet (modeled-duration) or
window `F·amt/R` (modeled-rate) **resolved once per cycle from the per-cycle PK jet** (not from the
unresolved `subject.doses`, `360-369`), then relax the `has_ss` arm at `532-540`. Modeled-rate SS
is the same moving-boundary class as the already-excluded rate-defined-SS-under-`F`
(`has_rate_defined_ss_infusion_under_f:430`) — scope carefully.

### C. zero_order + lagtime
Move the zero-order window onto the event-driven walk's lagtime saltation: `w_start = dose.time +
lag` becomes a moving boundary, so both window edges get `inject_rate_saltation` (rate-on at
`w_start` with `δlag`, rate-off at `w_end` with `δlag + δdur`). Relax `ode_analytical_supported:324`.

### D. zero_order + reset+input-rate
Admit `input_rate` + reset into the event-driven walk; replicate the static `reset_floor` guard
(`w_start >= reset_floor`, `3735`; `K_RESET` sets it at `3497`). Relax `ode_subject_supported:386`.

### E. zero_order + TV-cov
Route zero-order through the TV-cov event-driven walk (`integrate_tvcov_g` already carries the
rate-off saltation); relax `ode_tvcov_supported:514`. This depends on the TV-cov walk admitting
`input_rate` — coordinate with the non-IOV TV-cov work in `plans/ode-noniov-tvcov-combos.md`.

## Tests

Workhorses: `check_vs_production` (`4283`), `check_hessian_vs_fd_of_grad` (`4328`),
`check_iov_provider_vs_fd` (`provider.rs:5962`).

- **A**: new `ode_iov_modeled_duration_provider_matches_fd_of_predict_iov` (no such test exists
  today) — clone the IOV infusion test `ode_iov_infusion_provider_matches_fd_of_predict_iov`
  (`provider.rs:6331`) with a modeled `D1`; add a κ-on-`D1` variant.
- **B**: clone SS infusion test (`ode_provider.rs:5958`) with a modeled duration; keep a
  modeled-rate-SS still-FD edge test until B covers it.
- **C/D/E**: extend the #530 zero-order tests (`ode_provider_zero_order_absorption_matches_
  production:5384`, multi-dose `5442`) with lagtime / reset / TV-cov variants; flip the
  corresponding still-FD assertions.

## NONMEM comparison (required by CLAUDE.md)

For each combination, fit ferx vs NONMEM `METHOD=1 INTER` with `RATE=-2`/`D1` (and `SS=1`, IOV
`$OMEGA BLOCK`, reset/`ADDL`, or TV-cov as applicable); record OFV + parameter agreement in the PR.

## Docs + changelog

- Matrix issue **#486**: flip the ODE cells / notes on **Steady-state dosing**,
  **Modeled-duration/rate**, **Zero-order absorption**, **Time-varying covariates** rows; update
  Audit ② / ③ buckets (`167`, `173-175`).
- `CHANGELOG.md` `[Unreleased]` → `Added`. ferx-r: no follow-up unless a `pub` signature changes.

## Related plans

- `plans/rate-neg2-analytical.md` (#324) and `plans/analytical-zero-order-depot.md` (#400) extend
  the **closed-form** engine; they share the `RateMode`/`D{cmt}` plumbing and `resolve_rate`
  single-source-of-truth this ODE work reads, but scope no ODE-sensitivity combinations. #324's
  "SS infusions × `RATE=-2`" risk note (`115-116`) is the analytical mirror of design **B**.

## Risk / watch-outs

- **Inner/outer scope parity** is load-bearing: gates are shared by outer
  (`ode_subject_sensitivities_iov:2049`) and inner (`ode_subject_eta_grad_iov:2076`), and by the
  static-vs-tvcov split — keep them matched (`345-348`, `502-507`) or you get analytic-outer +
  FD-inner.
- **modeled-dose + SS per-cycle window**: resolve the duration jet *once per cycle* from the
  per-cycle PK jet (`2858-2862`), not from `subject.doses`.
- **`reset_floor` for zero-order**: a reset-cut window must still fire its rate-off correction
  (`3735`, `K_RESET` at `3497`).
- **Single-source dual + f64 walks** (CLAUDE.md): keep the boundary-break-time set and floor
  epsilons (`DURATION_FLOOR`/`RATE_FLOOR`, used `3081`/`3087`) identical to the f64 references
  (`active_zero_order_inputs`/`zero_order_windows`/`resolve_rate` in `ode/predictions.rs` /
  `pk/event_driven.rs`), or the saltation lands off the break.
- **Break-time placement**: the moving boundary must land on an exact timeline break (zero-order
  `3620-3625`, infusions `3616-3618`); push the corresponding `w_end`/`t_inf` break for any newly
  admitted combination so `inject_rate_saltation` fires on a real segment edge.
