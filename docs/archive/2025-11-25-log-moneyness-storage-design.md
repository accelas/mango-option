<!-- SPDX-License-Identifier: MIT -->
# Log-Moneyness Storage for Price Table

## Problem

The price table B-spline currently stores moneyness (m = S/K) on axis 0. This causes suboptimal interpolation because:

1. **Asymmetry**: Option prices are naturally symmetric in log-moneyness, not linear moneyness
2. **Grid efficiency**: Uniform moneyness grid is non-uniform in log-space, wasting resolution
3. **Interpolation error**: B-spline error is ~20-40% higher at tails with linear moneyness

## Solution

Store log-moneyness (x = ln(S/K)) internally while keeping the user-facing API in moneyness.

### Key Changes

**1. PriceTableSurface**
- `build()`: Transform axis 0 from moneyness to log-moneyness before B-spline fitting
- `value()`: Apply `std::log(coords[0])` before B-spline evaluation
- `partial()`: Apply log transform for queries

**2. PriceTableMetadata**
- Add `m_min`, `m_max` fields for original moneyness bounds (used for validation)

**3. PriceTableWorkspace**
- Store log-moneyness in the `moneyness_` field (rename to `log_moneyness_`)
- Add `m_min_`, `m_max_` to metadata
- Arrow IPC files will contain log-moneyness directly

**4. IVSolverInterpolated**
- Extract `m_range_` from metadata instead of axes
- No other changes needed (already queries in moneyness)

### Performance

- Query overhead: One `std::log()` call (~3ns) per evaluation
- B-spline eval: ~500ns
- Net impact: <1% overhead, better interpolation accuracy

### Compatibility

Breaking change for saved files. Old Arrow IPC files will not load correctly.
This is acceptable as nothing is in production yet.

## Files to Modify

1. `src/option/table/price_table_metadata.hpp` - Add m_min, m_max
2. `src/option/table/price_table_surface.hpp` - Add log transform in value/partial
3. `src/option/table/price_table_surface.cpp` - Transform axis 0 in build()
4. `src/option/table/price_table_workspace.hpp` - Rename to log_moneyness, add bounds
5. `src/option/table/price_table_workspace.cpp` - Update save/load for new fields
6. `src/option/iv_solver_interpolated.cpp` - Get m_range from metadata

## Testing

- Existing tests should pass (API unchanged)
- Add test comparing interpolation error: log vs linear moneyness
- Benchmark to confirm no performance regression
