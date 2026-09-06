# #485 — Correct fixed-expiry dividend sampling

Gate 1. Read [contract.md](contract.md), especially sections 2 and 7.
Prerequisites: #484 and the math-fix baseline #489. Refresh merge state first.

## Outcome

A segmented Chebyshev table, its query validation, and its reference oracles
all describe one fixed expiry across remaining life. Every raw snapshot has
the requested time and calendar-event side. Dividend padding and local-origin
offsets cannot silently change the priced contract.

## Start with RED

1. Build one fixed-expiry contract with a cash dividend. Compare several
   remaining-life queries before and after that event against direct solves
   using d_query = d_build - (T0 - tau). Include a point where the dividend
   has already elapsed but the old chain oracle would include it. Done when
   the current mismatch is reproduced and both competing contracts are
   printed/identified unambiguously.
2. Reproduce the 1.01 horizon timing offset with a direct fixed-expiry oracle.
   Check actual snapshot labels and both calendar-event sides. Done when the
   failure is attributed to time/event semantics rather than a fitted tensor.
3. Reproduce the routing-origin offset at a post-gap segment. Done when
   local coordinates derived by routing match those used during fitting.

## Implement within existing ownership

- Keep a single end-to-end fixed-expiry solve per K_ref/sigma/rate combination.
  Remove the padding-induced event skew; use the contractual horizon and
  exact requested sample times. Retain unrelated continuous-path padding
  only where its time-homogeneous semantics remain valid.
- Put anchored-to-query schedule conversion in one shared numerical helper
  used by segmented validation/oracles. Inputs are durations and the anchored
  dividend model, not a wall clock.
- Align query schedule admission with rolled offsets and post-dividend
  calendar-side semantics at an exact event.
- Preserve before/after event data explicitly where a segment endpoint needs
  them. A small internal snapshot/event extension is permissible if required
  for correctly labelled data. Preserve ordinary public pricing interfaces.
- Fit using local origins consistent with TauSegmentSplit. Keep K_ref and
  temporal split composition.
- Treat source-era chain-semantics per-node solve plans as superseded.

## Characterization and completion

Compare raw rows with independent direct solves on the same physical contract,
covering low sigma, short positive maturity, both puts/calls, event boundaries,
and deep tails. Establish reference convergence below the regression budget.
Preserve tighter existing bounds; report raw-sample and fitted-price errors
separately.

Remeasure existing Chebyshev coverage and documented adaptive-dividend pins
using the corrected fixed-expiry oracle. Gate 1 may retain documented
downstream fitting/refusal limitations; it does not require #458/#460 or a
comparative backend promotion to be solved first.

Complete when RED cases are GREEN, time labels/origins match, query/oracle
schedules agree, and relevant tests pass without relaxing their numerical
meaning. An exact boundary that cannot be represented yields an explicit
failure, not a neighboring-time price.

## Entry points and validation

Inspect:
- src/option/table/chebyshev/chebyshev_adaptive.cpp
- src/option/table/adaptive_metrics.cpp
- src/option/table/adaptive_refinement.cpp
- src/option/interpolated_iv_solver.hpp
- src/option/dividend_utils.hpp
- src/pde/core/grid.hpp and src/option/american_option_batch.cpp for exact-time plumbing

Run relevant Chebyshev cache, adaptive integration, factory, and snapshot
tests. Update the mathematical/interpolation/API documentation of segmented
tau semantics. Leave full continuous-path snapshot audits outside this task
unless a shared plumbing correction is required; report newly discovered
independent blockers explicitly.
