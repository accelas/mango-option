<!-- SPDX-License-Identifier: MIT -->
# Gamma Interpolation Implementation Design

**Date:** 2025-11-01
**Priority:** P3 from Issue #39
**Status:** Design Complete, Ready for Implementation

## Goal

Add gamma (∂²V/∂S²) computation and interpolation to price tables, following the vega implementation pattern from PR #49.

## Background

Vega interpolation (P1, PR #49) achieved 5.34% average accuracy with 8ns query performance. Gamma follows the same pattern but requires careful handling of coordinate transforms because it measures second-order price curvature.

## Why Gamma Matters

Traders use gamma for delta hedging and risk management. Accurate gamma is critical for:
- Rebalancing delta-neutral portfolios
- Measuring convexity risk
- Managing gamma scalping strategies

Second derivatives amplify numerical errors. We prioritize accuracy over speed.

## Architecture

### Data Structure

Add `double *gammas` to `OptionPriceTable` struct:

```c
typedef struct OptionPriceTable {
    // ... existing fields ...
    double *vegas;    // ∂V/∂σ (existing)
    double *gammas;   // ∂²V/∂S² (NEW)
} OptionPriceTable;
```

Lazy allocation follows vega pattern:
- NULL until `price_table_precompute()` allocates
- Same dimensions as prices (n_m × n_tau × n_sigma × n_r × [n_q])
- Initialize to NaN on allocation

### Computation Flow

Extend `price_table_precompute()` with third pass:

1. **First pass:** Compute prices via FDM (unchanged)
2. **Second pass:** Compute vegas ∂V/∂σ (unchanged)
3. **NEW - Third pass:** Compute gammas ∂²V/∂m²

### Coordinate Transform Handling

The critical innovation: proper COORD_LOG_SQRT handling.

When tables store log(m):
- Grid spacing is uniform in log-space: Δ(log m) = constant
- We need ∂²V/∂m², not ∂²V/∂(log m)²
- Chain rule transforms between spaces

**Mathematics:**

```
∂²V/∂m² = (1/m²)[∂²V/∂(log m)² - ∂V/∂(log m)]
```

Finite differences provide both terms:
```
∂²V/∂(log m)² ≈ (V[i+1] - 2V[i] + V[i-1]) / h²
∂V/∂(log m) ≈ (V[i+1] - V[i-1]) / 2h
```

This ensures gamma values are accurate in raw coordinates even when the stored grid uses log-transforms.

## Implementation Details

### Gamma Computation Algorithm

Handle three boundary cases for SIMD vectorization:

**Lower Boundary (i_m = 0) - Forward Difference:**

Use forward 3-point stencil when no left neighbor exists:

```c
if (coord_system == COORD_LOG_SQRT) {
    double m0 = exp(moneyness_grid[0]);
    double h = moneyness_grid[1] - moneyness_grid[0];

    double d2V = (V[2] - 2*V[1] + V[0]) / (h * h);
    double dV = (V[1] - V[0]) / h;

    gamma[0] = (d2V - dV) / (m0 * m0);
} else {
    double h = moneyness_grid[1] - moneyness_grid[0];
    gamma[0] = (V[2] - 2*V[1] + V[0]) / (h * h);
}
```

**Interior Points (1 ≤ i_m < n_m-1) - Centered Difference:**

```c
#pragma omp simd
for (size_t i_m = 1; i_m < n_m - 1; i_m++) {
    if (coord_system == COORD_LOG_SQRT) {
        double m = exp(moneyness_grid[i_m]);
        double h = moneyness_grid[i_m+1] - moneyness_grid[i_m];

        double d2V_dlogm2 = (V[i_m+1] - 2*V[i_m] + V[i_m-1]) / (h * h);
        double dV_dlogm = (V[i_m+1] - V[i_m-1]) / (2 * h);

        gamma[i_m] = (d2V_dlogm2 - dV_dlogm) / (m * m);
    } else {
        double h = moneyness_grid[i_m+1] - moneyness_grid[i_m];
        gamma[i_m] = (V[i_m+1] - 2*V[i_m] + V[i_m-1]) / (h * h);
    }
}
```

**Upper Boundary (i_m = n_m-1) - Backward Difference:**

Mirror of lower boundary using backward stencil.

**NaN Handling:**

Set result to NaN if any price in stencil is NaN. This matches vega's behavior.

### API Design

**Storage Access:**

```c
double price_table_get_gamma(const OptionPriceTable *table,
                             size_t i_m, size_t i_tau, size_t i_sigma,
                             size_t i_r, size_t i_q);

int price_table_set_gamma(OptionPriceTable *table,
                          size_t i_m, size_t i_tau, size_t i_sigma,
                          size_t i_r, size_t i_q, double gamma);
```

**Interpolation:**

```c
double price_table_interpolate_gamma_4d(const OptionPriceTable *table,
                                        double moneyness, double maturity,
                                        double volatility, double rate);

double price_table_interpolate_gamma_5d(const OptionPriceTable *table,
                                        double moneyness, double maturity,
                                        double volatility, double rate,
                                        double dividend);
```

### File Format Update

Bump version to 3, add gamma presence flag:

```c
typedef struct {
    uint32_t magic;        // 0x50545442 ("PTTB")
    uint32_t version;      // 3 (was 2)
    // ... existing fields ...
    uint8_t has_gammas;    // NEW: 1 if gammas present, 0 otherwise
    uint8_t padding[119];  // Reduced from 120
} PriceTableHeader;
```

**Save/Load Order:**
1. Header
2. Grid arrays
3. Prices
4. Vegas (if present)
5. Gammas (if present)

**Backward Compatibility:**

Version 2 files load with `gammas = NULL`. Loader skips gamma data if `has_gammas == 0`.

## Testing Strategy

### Unit Tests (tests/price_table_test.cc)

Add seven tests mirroring vega coverage:

1. **GammaArrayAllocation** - Verify lazy allocation
2. **GammaGetSet** - Test index-based access
3. **GammaPrecomputation** - Verify computation from prices
4. **GammaInterpolation4D** - Test 4D interpolation accuracy
5. **GammaInterpolation5D** - Test 5D with dividends
6. **GammaSaveLoad** - Verify binary persistence
7. **LoadOldFormatWithoutGamma** - Backward compatibility

### Accuracy Benchmark (benchmarks/gamma_accuracy.cc)

Create standalone benchmark following `vega_accuracy.cc`:

- Create moderate grid (5×4×5×3 = 300 points)
- Compute reference gamma via centered differences at off-grid points
- Compare with interpolated gamma
- Report average and maximum relative error

**Expected Results:**
- Average error: 5-10% (similar to vega's 5.34%)
- Maximum error: <20%
- ATM accuracy: <1% (where gamma matters most)

### Integration Tests

Update `accuracy_comparison.cc`:
- Compare FDM-computed gamma vs interpolated gamma
- Verify chain rule transformation works correctly
- Test both COORD_RAW and COORD_LOG_SQRT modes

## Performance Targets

**Precomputation:**
- Add 20-30 seconds to current 3-minute baseline
- ~10% overhead (acceptable for one-time cost)
- Throughput: ~5,000 gammas/second

**Query:**
- Target: ~8ns per gamma (matching vega)
- Uses same cubic spline interpolation infrastructure
- No performance regression vs vega queries

**Accuracy:**
- Average relative error: <10%
- ATM relative error: <1%
- Production-ready for hedging applications

## Implementation Checklist

**Core Changes:**
- [ ] Add `double *gammas` field to `OptionPriceTable`
- [ ] Implement gamma computation in `price_table_precompute()` (third pass)
- [ ] Handle COORD_LOG_SQRT chain rule transformation
- [ ] Implement boundary cases (forward/centered/backward differences)
- [ ] Add SIMD vectorization pragma to interior loop

**API Functions:**
- [ ] Implement `price_table_get_gamma()`
- [ ] Implement `price_table_set_gamma()`
- [ ] Implement `price_table_interpolate_gamma_4d()`
- [ ] Implement `price_table_interpolate_gamma_5d()`

**File I/O:**
- [ ] Bump version to 3
- [ ] Add `has_gammas` flag to header
- [ ] Update `price_table_save()` to write gammas
- [ ] Update `price_table_load()` to read gammas
- [ ] Verify backward compatibility with v2 files

**Testing:**
- [ ] Add 7 unit tests to `tests/price_table_test.cc`
- [ ] Create `benchmarks/gamma_accuracy.cc`
- [ ] Update `benchmarks/BUILD.bazel`
- [ ] Run all tests and verify pass

**Documentation:**
- [ ] Update `src/price_table.h` header comments
- [ ] Add gamma example to file header
- [ ] Update `CLAUDE.md` with gamma workflow
- [ ] Update issue #39 with P3 status

## Success Criteria

- [ ] All unit tests pass
- [ ] Gamma accuracy: <10% average error
- [ ] ATM accuracy: <1%
- [ ] Query performance: ~8ns per gamma
- [ ] Precomputation overhead: <15%
- [ ] Backward compatibility: v2 files load correctly
- [ ] Integration tests show agreement with FDM-computed gamma

## References

- PR #49: Vega interpolation implementation (pattern to follow)
- Issue #39: Interpolation accuracy improvements roadmap
- `benchmarks/vega_accuracy.cc`: Template for gamma accuracy test

## Notes

Second derivatives amplify numerical errors. The chain rule transformation for COORD_LOG_SQRT is critical for accuracy. Test thoroughly with both coordinate systems.
