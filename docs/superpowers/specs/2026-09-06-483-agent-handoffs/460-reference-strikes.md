# #460 — Reference-strike semantics and automatic selection

Gate 4. Read [contract.md](contract.md), especially sections 3–5.
Depends on #488. Sparse-K_ref characterization from #462 is performed here,
before final benchmark retuning.

## Outcome

Manual and adaptive segmented surfaces evaluate the same physical query,
enforce the same supported strike/moneyness domain, and use one consistent
reference selection/validation policy.

## Sequence

1. Add RED price and Greek tests through public surface/query interfaces.
   Test exact reference strikes, midpoints, endpoints, moving spot, and
   queries having equal S/K but different absolute K. Done when tests expose
   the current mapping/coverage defects without mixing in chained samples.
2. Correct local spot mapping to S*K_i/K and retain linear-in-strike positive
   weights. Propagate delta/gamma chain factors through the split seam;
   vega/rho/theta do not gain a spot-map factor. Done when public Greeks
   agree with independent finite differences on the same composed surface
   and with appropriate direct-price references.
3. Define/publish the absolute strike interval alongside log-moneyness and
   other query bounds. Derive it from the requested option domain. Enforce
   checked query admission; document any intentionally unchecked primitive
   as requiring an in-domain query. Done when bounds cannot silently widen
   after composition or loading.
4. Consolidate reference resolution across manual/adaptive builders. Automatic
   coverage derives from the requested strike interval; automatic density
   follows measured criteria within ceilings. Legacy count/span heuristics
   cannot truncate that interval. Explicit reference arrays retain their
   exact validated values.
5. Characterize reference density using corrected samples. Use K endpoints,
   off-K midpoints, both moneyness tails, low sigma, remaining-life positions
   on both sides of dividends, multiple dividend amounts, and both backends.
   Vary spot independently of strike. Done when accepted/rejected density
   behavior is supported by reproducible measurements, not one universal
   spacing guessed from the old broken fit.
6. Update persistence for the supported strike bounds and necessary
   fixed-expiry numerical metadata; version/reject incomplete legacy data
   according to the shared contract.

## Acceptance

No invalid explicit set is silently repaired or replaced. An inadequate fixed
set fails the applicable criteria. Automatic selection stops at its declared
limits and reports failure when no acceptable set is found.

Run split_surface, segmented surface/builder, adaptive integration,
factory, price_table_data, and parquet tests. Correct the conflicting
linear/log-strike/Catmull-Rom prose. Record K density, measured price/IV
errors, filtered/refused counts, and build/query cost. Do not retune #462's
final benchmark population here.

Inspect splits/multi_kref.hpp, split_surface.hpp, adaptive_refinement.cpp,
bspline_adaptive.cpp, both segmented builders, price_table_factory.cpp,
and serialization/parquet code.
