<!-- SPDX-License-Identifier: MIT -->
# Cache Locality and Coordinate Transform Optimization

**Date:** 2025-10-30
**Status:** Approved for Implementation
**Related Issues:** #39 (Coordinate Transform), #40 (Unified Optimization)

## Problem Statement

Two independent problems degrade interpolation performance and accuracy:

1. **Cache-unfriendly memory layout**: Moneyness slice extraction requires 30KB stride jumps (3750 elements), causing 100x cache miss rate compared to optimal layout
2. **Coordinate system instability**: Interpolating in raw (m, T, σ) space produces 26% average error and 151% maximum error

These problems are orthogonal and can be solved together without conflicts.

## Proposed Solution

### Architecture: Layered Transformation Pipeline

User queries flow through four independent layers:

```
User Query: price_table_interpolate_4d(table, m=1.05, T=0.5, σ=0.25, r=0.03)
  ↓
Layer 1: Coordinate Transform → (log(m)=0.0488, sqrt(T)=0.707, σ=0.25, r=0.03)
  ↓
Layer 2: Grid Lookup → find_bracket() → (i_m=15, i_tau=8, i_sigma=5, i_r=3)
  ↓
Layer 3: Index Calculation → Apply strides → idx = 15*stride_m + 8*stride_tau + ...
  ↓
Layer 4: Memory Access → prices[idx]
```

**Key principle**: Each layer is stateless and independently testable. Layer 1 improves accuracy, Layer 3 improves cache performance.

### Data Structure Changes

Add two metadata fields to `OptionPriceTable`:

```c
typedef enum {
    COORD_RAW,           // Current: m, T, σ, r, q (no transformation)
    COORD_LOG_SQRT,      // Recommended: log(m), sqrt(T), σ, r, q
    COORD_LOG_VARIANCE,  // Advanced: log(m), σ²T, r, q (future)
} CoordinateSystem;

typedef enum {
    LAYOUT_M_OUTER,      // Current: [m][tau][sigma][r][q] (stride_m = 3750)
    LAYOUT_M_INNER,      // Optimized: [r][sigma][tau][m] (stride_m = 1)
    LAYOUT_BLOCKED,      // Future: cache-oblivious tiled layout
} MemoryLayout;

typedef struct OptionPriceTable {
    // Existing fields unchanged
    size_t n_moneyness, n_maturity, n_volatility, n_rate, n_dividend;
    double *moneyness_grid;  // Grid values (interpretation depends on coord_system)
    double *maturity_grid;   // Grid values (interpretation depends on coord_system)
    double *prices;          // Flattened array (layout depends on memory_layout)

    // New metadata
    CoordinateSystem coord_system;  // How to interpret grid values
    MemoryLayout memory_layout;     // How prices are stored in memory

    // Strides (computed based on memory_layout)
    size_t stride_m, stride_tau, stride_sigma, stride_r, stride_q;
} OptionPriceTable;
```

**Semantic note**: `moneyness_grid` stores transformed coordinates (e.g., log(m) values if `COORD_LOG_SQRT`). User API accepts raw coordinates; transformation happens internally.

### API Design

#### Backward-Compatible Creation

```c
// Existing API unchanged - defaults to current behavior
OptionPriceTable* price_table_create(
    const double *moneyness, size_t n_m,
    // ... existing parameters
    OptionType type, ExerciseType exercise);
// Defaults: coord_system = COORD_RAW, memory_layout = LAYOUT_M_OUTER
```

#### Extended Creation API

```c
// New API with explicit transformation control
OptionPriceTable* price_table_create_ex(
    const double *moneyness, size_t n_m,
    // ... existing parameters
    OptionType type, ExerciseType exercise,
    CoordinateSystem coord_system,     // NEW
    MemoryLayout memory_layout);       // NEW
```

#### Query API (Unchanged Interface)

```c
// User queries with raw coordinates
double price = price_table_interpolate_4d(table, 1.05, 0.5, 0.25, 0.03);
// Internally transforms to (log(1.05), sqrt(0.5)) before find_bracket()
```

**Design rationale**: Users think in raw coordinates, so API accepts raw values. Transformation is an implementation detail.

#### Slice Extraction API (New)

```c
typedef enum {
    SLICE_DIM_MONEYNESS, SLICE_DIM_MATURITY, SLICE_DIM_VOLATILITY,
    SLICE_DIM_RATE, SLICE_DIM_DIVIDEND
} SliceDimension;

/**
 * Extract 1D slice along specified dimension
 *
 * @param dimension: Which dimension to extract
 * @param fixed_indices: Values for other dimensions (array[5], use -1 to vary)
 * @param out_slice: Output buffer (user-provided)
 * @param is_contiguous: [OUT] True if zero-copy possible, false if copied
 * @return 0 on success, -1 on error
 *
 * Example: Extract moneyness slice at (tau=5, sigma=3, r=2)
 *   int fixed[] = {-1, 5, 3, 2, 0};
 *   price_table_extract_slice(table, SLICE_DIM_MONEYNESS, fixed, slice, &contiguous);
 */
int price_table_extract_slice(
    const OptionPriceTable *table,
    SliceDimension dimension,
    const int *fixed_indices,
    double *out_slice,
    bool *is_contiguous);
```

#### Advanced Raw Buffer Access

```c
// Expert-only: direct buffer access (layout-dependent)
const double* price_table_get_raw_buffer(const OptionPriceTable *table);
void price_table_get_strides(const OptionPriceTable *table, ...);
```

### Implementation Details

#### Coordinate Transformation (Layer 1)

```c
static void transform_query_to_grid(
    CoordinateSystem coord_system,
    double m_raw, double tau_raw, double sigma_raw, double r_raw,
    double *m_grid, double *tau_grid, double *sigma_grid, double *r_grid)
{
    switch (coord_system) {
        case COORD_RAW:
            *m_grid = m_raw;
            *tau_grid = tau_raw;
            break;
        case COORD_LOG_SQRT:
            *m_grid = log(m_raw);
            *tau_grid = sqrt(tau_raw);
            break;
    }
    *sigma_grid = sigma_raw;
    *r_grid = r_raw;
}
```

#### Stride Calculation (Layer 3)

```c
static void compute_strides(OptionPriceTable *table) {
    size_t n_m = table->n_moneyness;
    size_t n_tau = table->n_maturity;
    size_t n_sigma = table->n_volatility;
    size_t n_r = table->n_rate;
    size_t n_q = table->n_dividend > 0 ? table->n_dividend : 1;

    switch (table->memory_layout) {
        case LAYOUT_M_OUTER:  // Current: [m][tau][sigma][r][q]
            table->stride_m = n_tau * n_sigma * n_r * n_q;
            table->stride_tau = n_sigma * n_r * n_q;
            table->stride_sigma = n_r * n_q;
            table->stride_r = n_q;
            table->stride_q = 1;
            break;

        case LAYOUT_M_INNER:  // Optimized: [q][r][sigma][tau][m]
            table->stride_q = n_r * n_sigma * n_tau * n_m;
            table->stride_r = n_sigma * n_tau * n_m;
            table->stride_sigma = n_tau * n_m;
            table->stride_tau = n_m;
            table->stride_m = 1;  // Moneyness contiguous
            break;
    }
}
```

**Key insight**: With `LAYOUT_M_INNER`, moneyness slice extraction becomes a single `memcpy()` instead of strided loop.

#### Slice Extraction Implementation

```c
int price_table_extract_slice(
    const OptionPriceTable *table,
    SliceDimension dimension,
    const int *fixed_indices,
    double *out_slice,
    bool *is_contiguous)
{
    size_t slice_stride, slice_length;

    // Determine stride and length
    switch (dimension) {
        case SLICE_DIM_MONEYNESS:
            slice_stride = table->stride_m;
            slice_length = table->n_moneyness;
            break;
        // ... other dimensions
    }

    // Calculate base offset
    size_t base_idx = 0;
    if (fixed_indices[0] >= 0) base_idx += fixed_indices[0] * table->stride_m;
    if (fixed_indices[1] >= 0) base_idx += fixed_indices[1] * table->stride_tau;
    // ... other indices

    // Extract: zero-copy if contiguous, strided copy otherwise
    if (slice_stride == 1) {
        *is_contiguous = true;
        memcpy(out_slice, &table->prices[base_idx], slice_length * sizeof(double));
    } else {
        *is_contiguous = false;
        for (size_t i = 0; i < slice_length; i++) {
            out_slice[i] = table->prices[base_idx + i * slice_stride];
        }
    }
    return 0;
}
```

### File Format Versioning

```c
typedef struct PriceTableFileHeader {
    uint32_t magic;              // 0x50524943 ("PRIC")
    uint16_t version;            // Bump to 2
    uint16_t coord_system;       // NEW: serialize CoordinateSystem
    uint16_t memory_layout;      // NEW: serialize MemoryLayout
    // ... existing fields
} PriceTableFileHeader;
```

**Backward compatibility**: Version 1 files interpreted as `COORD_RAW` + `LAYOUT_M_OUTER`.

### Migration Strategy

**Phase 1**: Implement transformations with current behavior as default
- All tests pass with existing code
- New APIs available but not required

**Phase 2**: Update benchmarks to measure improvements
- Add `--coord-system` and `--memory-layout` flags
- Demonstrate 10x accuracy + 6x speed gains

**Phase 3**: Migrate production tables gradually
- Convert one table at a time using migration tool
- Validate accuracy matches old tables

### Testing Strategy

#### Unit Tests

- Coordinate transform round-trip accuracy
- Stride calculation for each layout
- Index mapping consistency (unflatten→flatten is identity)

#### Integration Tests

- Verify `COORD_LOG_SQRT` improves accuracy vs `COORD_RAW`
- Verify all layouts produce identical interpolation results
- File format versioning compatibility

#### Performance Benchmarks

- Slice extraction speed: `LAYOUT_M_OUTER` vs `LAYOUT_M_INNER`
- Full interpolation throughput comparison
- Cache miss measurement via `perf stat`

## Expected Impact

| Optimization | Accuracy | Speed | Memory |
|--------------|----------|-------|--------|
| Log-moneyness transform | 5x better | — | — |
| Dimension reordering | — | 3x faster | — |
| Non-uniform grids (future) | 2x better | — | 10x less |
| Vega interpolation (future) | Greeks accurate | 2x faster IV | 2x more |
| **COMBINED** | **10x better** | **6x faster** | **5x less** |

## Files to Modify

### Phase 1: Core Infrastructure
- `src/price_table.h` - Add enums and fields
- `src/price_table.c` - Implement transformations, stride calculation
- `src/iv_surface.h` - Add coordinate system support
- `src/iv_surface.c` - Apply to 2D case

### Phase 2: Interpolation Updates
- `src/interp_multilinear.c` - Add coordinate transform
- `src/interp_cubic.c` - Add coordinate transform, use slice API
- `src/interp_strategy.h` - Document layout considerations

### Phase 3: Testing
- `tests/coordinate_transform_test.cc` - New unit tests
- `tests/memory_layout_test.cc` - New unit tests
- `tests/interpolation_accuracy_test.cc` - Update with new configs
- `benchmarks/cache_benchmark.cc` - New cache profiling benchmark

## Success Criteria

- [ ] Interpolation error < 1 bp in implied volatility for 95% of queries
- [ ] Slice extraction 3-5x faster with `LAYOUT_M_INNER`
- [ ] Cache miss rate drops 100x (measured via perf)
- [ ] All tests pass with both old and new configurations
- [ ] Backward compatibility maintained (old behavior is default)
- [ ] File format versioning prevents loading incompatible tables

## References

- Issue #39: Coordinate transform recommendations
- Issue #40: Unified cache + coordinate optimization
- "What Every Programmer Should Know About Memory" - Ulrich Drepper
- Strunk & White: Active voice, definite language, omit needless words
