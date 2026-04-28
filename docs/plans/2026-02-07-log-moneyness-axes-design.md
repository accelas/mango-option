# Design: Store axes.grids[0] as log-moneyness natively

**Issue:** #373
**Date:** 2026-02-07
**Branch:** `feature/log-moneyness-axes`

## Problem

`PriceTableAxes::grids[0]` stores moneyness (S/K) but every consumer
immediately converts to log-moneyness. The `log()` transform is applied
redundantly in 6 places, and chain rule corrections are needed in
`partial()` and `second_partial()`.

## Design

### 1. Core — PriceTableAxes and PriceTableSurface

`axes.grids[0]` stores `log(S/K)`. Axis name changes to
`"log_moneyness"`. Metadata `m_min`/`m_max` store log-moneyness.

Deletions in PriceTableSurface:
- **build()** — Remove log-transform loop. grids[0] passes straight to
  B-spline. m_min/m_max set from grids[0].front()/.back() directly.
- **value()** — Remove `log(coords[0])`. Caller passes log-moneyness.
- **partial()** — Remove `log()` transform and `/ m` chain rule.
  Return `spline_->eval_partial()` directly.
- **second_partial()** — Remove `log()` transform and
  `(g'' - g') / m²` chain rule. Return `spline_->eval_second_partial()`
  directly.

Deletions in PriceTableBuilder:
- **extract_tensor()** — Remove `log_moneyness` precomputation. Use
  `axes.grids[0]` directly for spline eval.
- **fit_coeffs()** — Remove moneyness-to-log transform loop.

Net: 6 `log()` transforms and 2 chain rule corrections deleted.

### 2. AmericanPriceSurface

Pass `log(spot/strike)` or `log(spot/K_ref)` as coords[0] instead of
`spot/strike`.

**price()** — Compute `x = std::log(spot / K_ref)` or
`std::log(spot / strike)`, pass as coords[0]. No other changes.

**delta()** — `partial(0, ...)` now returns `∂EEP/∂x` (log-moneyness
derivative). Apply chain rule here:
```
delta_eep = (1/K_ref) * (∂EEP/∂x) / m
          = (1/K_ref) * (∂EEP/∂x) * (strike / spot)
```

**gamma()** — `second_partial(0, ...)` returns `∂²EEP/∂x²`. Apply
full second-order chain rule here:
```
gamma_eep = (K / K_ref) * (∂²EEP/∂x² - ∂EEP/∂x) / (m² * K²)
```

The chain rule moves from PriceTableSurface (generic) to
AmericanPriceSurface (knows physical context). This is the right place.

**m_min() / m_max()** — Return log-moneyness from metadata directly.

### 3. Factory methods

**from_vectors()** — Accept log-moneyness directly. Parameter renamed
from `moneyness` to `log_moneyness`. Positivity check removed (log
values can be negative). Sort-and-dedupe unchanged (log preserves
ordering).

**from_strikes()** — Compute `log(spot / K)` instead of `spot / K`.
This is the natural conversion boundary.

**from_grid() / from_grid_auto()** — Feed into from_vectors() /
from_strikes(). `estimate_grid_from_grid_bounds()` produces
log-moneyness instead of moneyness.

### 4. PriceSurface concept and spliced surfaces

`m_min()` / `m_max()` return log-moneyness. The concept signature is
unchanged (still returns `double`); the semantic contract changes.

All types satisfying PriceSurface update:
- AmericanPriceSurface
- MultiKRefSurfaceWrapper
- StrikeSurfaceWrapper

SplicedSurfaceWrapper::Bounds stores log-moneyness.
InterpolatedIVSolver compares `log(query_m)` against bounds.

### 5. Persistence (Arrow IPC)

Stubbed out temporarily. PriceTableWorkspace save/load disabled so it
compiles but does not function. Persistence redesign deferred to a
follow-up issue.

### 6. Python bindings

Workspace creation parameters update to log-moneyness. Property
docstrings update. Workspace persistence tests disabled alongside
Arrow changes.

### 7. Tests

Mechanical updates to ~15 assertions:
- Coordinates passed to value()/partial()/second_partial() become log(m)
- m_min/m_max assertions become log-space values
- Workspace tests disabled with persistence

**Key invariant:** End-to-end prices, deltas, and gammas from
AmericanPriceSurface must not change. Existing tolerances serve as
the regression suite.

## Files affected

- `src/option/table/price_table_surface.cpp` — core simplification
- `src/option/table/price_table_surface.hpp` — doc comments
- `src/option/table/price_table_builder.cpp` — delete transforms
- `src/option/table/price_table_builder.hpp` — parameter rename
- `src/option/table/price_table_metadata.hpp` — doc update
- `src/option/table/price_table_axes.hpp` — axis name
- `src/option/table/american_price_surface.cpp` — chain rule moves here
- `src/option/table/american_price_surface.hpp` — doc update
- `src/option/table/spliced_surface.hpp` — bounds semantics
- `src/option/table/price_table_workspace.cpp` — stub persistence
- `src/option/table/price_table_workspace.hpp` — stub persistence
- `src/option/table/adaptive_grid_builder.cpp` — log-space metadata
- `src/option/interpolated_iv_solver.cpp` — log-space bounds
- `src/option/interpolated_iv_solver.hpp` — log-space bounds
- `src/python/mango_bindings.cpp` — parameter/docstring update
- `tests/price_table_surface_test.cc` — coordinate updates
- `tests/american_price_surface_test.cc` — bounds updates
- `tests/interpolated_iv_solver_test.cc` — bounds updates
- `tests/price_table_workspace_test.cc` — disable
- `tests/test_bindings.py` — disable workspace tests
