# #463 — Continuous Chebyshev adaptive factory

Gate 2. Read [contract.md](contract.md).
Depends on #480/#484; independent of #485/#488. This is separate from the ABI task.

## Outcome

A continuous Chebyshev factory request with adaptive configuration executes
the existing adaptive builder and publishes its diagnostics and validated
sample domain. Fixed num_pts applies to manual construction.

## Sequence and acceptance

1. Add a factory-level RED case demonstrating that adaptive configuration is
   currently ignored. Observe diagnostics, target/refusal behavior, and
   published bounds; avoid timing or exact-node-count assertions.
2. Map the factory's requested option domain to the existing adaptive
   Chebyshev builder. Preserve documented continuous model restrictions,
   option type, dividend yield, and maturity range. Done when all supplied
   adaptive controls reach the builder and its errors propagate faithfully.
3. Attach the returned diagnostics and sample bounds through the same
   ownership path used by other factory backends. Done when a query can
   neither access unvalidated support headroom nor lose build diagnostics.
4. Keep manual configuration behavior covered. Done when num_pts remains
   effective for manual builds and does not silently cap adaptive refinement.

Inspect price_table_factory.cpp, chebyshev_adaptive.cpp/.hpp,
interpolated_iv_solver.cpp/.hpp, and iv_solver_factory_test.cc.
Run factory, Chebyshev surface/adaptive, and error-mapping tests.

Coordinate hooks with later certification/acceptance gates when rebasing.
Do not implement the C ABI revision here. Do not close #463 until its ABI
half is also complete.
