# #461 — Preserve refinement positions under point ceilings

Gate 5. Read [contract.md](contract.md). Depends on #458 and #460.
Use #488's final fixed-expiry sampling/time interface.

## Outcome

The final segmented surface and retries use selected refined coordinates,
not merely the sizes of probe grids. This includes temporal placement
supported by the completed sampling interface.

## Sequence and RED cases

1. Trace all four axes from probe refinement through final assembly and retry.
   Temporal positions currently disappear before aggregation when converted
   to a point count. Done when the regression covers that earlier loss as
   well as aggregate_max_sizes/linspace reconstruction.
2. Add an observable pricing/accuracy regression where a retained refined grid
   meets its criterion but the sizes-only rebuild does not. Keep the PDE
   references fixed. Done when the failure measures lost placement rather
   than changing oracle data.
3. Carry actual vectors through assembly. Preserve temporal regimes and
   event-side semantics; merging a tau axis must not fit across a dividend.
4. Use the exact union when it fits the cap. On overflow, preserve endpoints
   and required seed/event nodes and select deterministic bounded candidates
   from actual refined positions. Evaluate against the same reference data.
   Refuse if no capped candidate is viable.
5. Apply the same placement policy to retries and skipped-probe handling.

## Completion

Tests cover deterministic results, within-cap and over-cap unions, retained
nonuniform positions, all relevant axes, retries, skipped probes, and
dividend boundaries. Public accuracy is assessed on the returned candidate.
No theorem of monotonically improving accuracy with increasing node count
is claimed.

Inspect adaptive_refinement.hpp/.cpp and bspline_adaptive.cpp; verify the
segmented builder accepts the actual tau coordinates rather than only a
count. Run adaptive refinement/grid-builder, segmented builder, and
adaptive-surface integration tests.
