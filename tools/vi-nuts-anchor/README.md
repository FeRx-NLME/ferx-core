# Anchor C — per-subject NUTS reference

The harness for **Anchor C** of `VI_VALIDATION.md` (§5, results in §5.1): the only anchor that
measures VI's *approximation* rather than its optimizer.

Every other reference in that document is Gaussian. AGQ's Laplace `H⁻¹` is a Gaussian at the
mode, so comparing `q` against it asks *did VI find the right Gaussian?* NUTS samples the true
posterior shape, so it asks *is a Gaussian right at all?* — which is the question
`docs/estimation/vi.qmd` publishes a number for.

## Running it

```bash
tools/vi-nuts-anchor/run.sh
```

First run builds an isolated venv at `$FERX_VI_STATE/pyenv` and installs jax + numpyro into it.
The system Python is never touched and deleting `$FERX_VI_STATE` undoes the install — the same
arrangement `tools/vi-emvi-comparison` uses for its R library. Built from
`/opt/anaconda3/bin/python3.11` (override with `FERX_PY`): the system `python3` here is 3.14,
which has no `numpy` and no jax wheels.

Not wired into CI, and shouldn't be: no CI image here carries jax, and NUTS takes minutes.

## Contents

| File | Role |
|---|---|
| `run.sh` | driver — provisions the venv, runs the ferx side, then the reference |
| `anchor_c.py` | numpyro model, NUTS, the Laplace reference, and the comparison |
| `q_at_agq.ferx` | ferx VI with every population parameter FIXed at the AGQ estimate |

## Why both sides are pinned at the AGQ estimate

The comparison is about `q`, so both sides must sit at the same `(θ, Ω, σ)` or the difference
mixes the approximation with a parameter difference. `q_at_agq.ferx` FIXes all of them at the AGQ
(`n_agq = 9`) values, which reproduce both ferx's and nlmixr2's FOCEI `σ` to six decimals.

That configuration was unusable until the convergence fix recorded in `VI_VALIDATION.md` §10:
with every population parameter FIXed, VI's parameter-stability criterion compared a constant
vector against itself and stopped the fit after 500 iterations. It now runs to a genuinely
converged `q` — `−2·ELBO = −285.924` against AGQ's `−2 log L = −285.977`, a gap of 0.05 units.

## Reading the output

Three ratio rows, and the middle one is the interpretive key:

- `VI var / NUTS var` — the understatement. **This is the published claim.**
- `Laplace var / NUTS var` — how Gaussian the true posterior actually is. At `1.000` there is
  nothing for a Gaussian `q` to get wrong, and a small number in the row above means the dataset
  cannot test the claim rather than that VI is unusually good.
- `VI var / Laplace var` — whether `q` found the right Gaussian, which is what every other anchor
  in the document is limited to asking.

`FERX_DATA` / `FERX_Q_FILE` / `FERX_ANCHOR_C_OUT` run the same comparison on another dataset; §5.1
uses them for a 2-observations-per-subject variant of warfarin.

## Deliberately not ferx's own `method = bayes`

It shares the likelihood code, which makes it an internal consistency check rather than an anchor.
