# #487 — Grid policy on the resolved grid

Non-blocking cleanup. Read [contract.md](contract.md).
Baseline includes #484's coverage ownership and #489's automatic batch
coverage fix. This task does not gate the core chain; completing it before
final tuning is useful.

## Outcome

Grid decisions assess the grid actually used, retain required coverage,
respect explicit constraints/ceilings, and report numerical limitations.
A width-only constant does not become an unsupported universal validity rule.

## Sequence

1. Inventory automatic accuracy requests, explicit grids, shared and
   normalized routes, and existing fallback behavior. Add public RED cases
   for mismatches between the grid judged and the grid solved, and for
   ceiling violations such as 5000 becoming 5001 after odd adjustment.
2. Characterize actual wide-grid behavior. MAX_WIDTH=5.8 is a routing
   heuristic; the core already supports explicit width-6 grids. Route or
   reject using justified numerical/accuracy criteria. Preserve required
   coverage instead of clipping the domain.
3. Repeat clearance characterization for calls as well as puts, including
   low sigma, wide moneyness, and relevant dividend boundaries. Retain the
   existing default clearance unless measurements justify changing it.
4. Compare default clustering derived after coverage folding against the
   current fixed-alpha behavior. Default alpha may be recomputed from the
   resolved geometry when justified; explicit alpha remains a constraint.
   An API representation must distinguish automatic choice from an expert
   override rather than guessing that a numeric value was a default.
5. Make spatial point ceilings strict, selecting a valid odd count within the
   ceiling. Reject inconsistent min/max requests explicitly.

Complete when actual-grid routing/coverage, explicit overrides, both option
types, and strict ceilings have focused regressions. Report accuracy,
resolution/cap hits, steps, and cost before/after. Update stale helper and
stability-limit prose.

Inspect grid_spec_types.cpp/.hpp, american_option_batch.cpp,
bspline_builder.cpp fallback routes, grid_spec_types_test,
american_option_batch_test, and custom-grid/coverage tests. Keep automatic
maturity-cohort selection outside scope.
