# #486 — Manual Chebyshev default accuracy

Non-blocking cleanup. Read [contract.md](contract.md).
Measure after #485's timeline/oracle correction. This task does not gate the
core dependency chain.

## Outcome

The manual segmented Chebyshev defaults have a truthful accuracy regression
on their actual requested domain. Explicit manual levels remain constraints.

## Sequence

1. Reproduce the issue on the corrected fixed-expiry path. Record the actual
   default levels, generated node counts, support/query bounds, raw PDE error,
   and off-node fit error. The issue's nine-moneyness-node explanation is
   stale relative to the currently recorded default level 5 (33 CC nodes).
2. Establish direct-oracle convergence below the applicable price tolerance.
   Test tails, exercise transitions, low sigma, and event-side positions.
   Done when sampling and fitting errors are separated.
3. If corrected defaults already meet the criterion, tighten the regression
   and correct the stale explanation; no density change is required.
   Otherwise change default density/headroom only with measured justification
   under the shared default/explicit-setting policy.
4. Verify explicit levels are honored and inadequate requests fail their
   requested criteria rather than silently becoming a different manual grid.

Complete when the actual defaults and off-node accuracy are pinned, price/
Greek/IV limitations are reported honestly, and relevant factory/adaptive
integration tests pass. Automatic maturity cohorts and a general new
interpolation framework are outside scope.

Inspect ChebyshevSegmentedBuilder defaults/build, the manual factory route,
and SegmentedChebyshevTailsMatchFdmAtExtremeMoneyness in
adaptive_surface_build_integration_test.cc.
