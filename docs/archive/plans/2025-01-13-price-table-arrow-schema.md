# Price Table Arrow IPC Schema Design

**Status:** Design Phase
**Created:** 2025-01-13
**Author:** Claude Code

## Executive Summary

Define Apache Arrow IPC format for persisting pre-computed option price tables with zero-copy mmap loading. Target: <1ms load time per table for intraday trading systems.

## Use Case

**Workflow:**
```
Overnight Batch (3am-6am):
  ├─ Compute 50-200 option surfaces
  ├─ Each table: ~3 minutes (200 PDE solves)
  └─ Save to Arrow IPC files

Pre-Market Load (8am):
  ├─ mmap all tables into memory
  ├─ Target: <100ms total (1ms per table)
  └─ Validate checksums, build spans

Trading Day (9:30am-4pm):
  ├─ Fast IV lookups: ~135ns per query
  ├─ Occasional reloads if vol spikes
  └─ Zero allocation after startup
```

**Critical Requirements:**
- ✅ Zero-copy mmap (no deserialization)
- ✅ Self-describing (schema embedded)
- ✅ Validated loads (checksums, version checks)
- ✅ Sub-millisecond load time
- ✅ Portable across x86_64 machines (same endianness)

## Arrow Schema Definition

### Schema Version 1.0

```python
# Arrow schema (using pyarrow notation)
PriceTableSchema = pa.schema([
    # ========================================================================
    # METADATA (Scalar fields)
    # ========================================================================
    pa.field("format_version", pa.uint32()),         # Schema version (1)
    pa.field("created_timestamp", pa.timestamp('us')), # UTC creation time
    pa.field("mango_version", pa.string()),          # Library version (git hash)

    # ========================================================================
    # OPTION PARAMETERS
    # ========================================================================
    pa.field("ticker", pa.string()),                 # Underlying symbol (e.g., "SPY")
    pa.field("option_type", pa.uint8()),             # 0=PUT, 1=CALL
    pa.field("K_ref", pa.float64()),                 # Reference strike price
    pa.field("dividend_yield", pa.float64()),        # Continuous dividend yield

    # ========================================================================
    # GRID DIMENSIONS (Scalar metadata)
    # ========================================================================
    pa.field("n_moneyness", pa.uint32()),            # Number of moneyness points
    pa.field("n_maturity", pa.uint32()),             # Number of maturity points
    pa.field("n_volatility", pa.uint32()),           # Number of volatility points
    pa.field("n_rate", pa.uint32()),                 # Number of rate points

    # ========================================================================
    # GRID VECTORS (1D arrays, mmap-able)
    # ========================================================================
    pa.field("moneyness", pa.list_(pa.float64())),   # Sorted ascending, ≥4 points
    pa.field("maturity", pa.list_(pa.float64())),    # Years, sorted ascending
    pa.field("volatility", pa.list_(pa.float64())),  # Annual vol, sorted ascending
    pa.field("rate", pa.list_(pa.float64())),        # Risk-free rate, sorted ascending

    # ========================================================================
    # KNOT VECTORS (Precomputed, clamped cubic B-spline)
    # ========================================================================
    # Note: Clamped cubic B-splines have n+4 knots (4 clamped at each end)
    pa.field("knots_moneyness", pa.list_(pa.float64())),   # Size: n_m + 4
    pa.field("knots_maturity", pa.list_(pa.float64())),    # Size: n_tau + 4
    pa.field("knots_volatility", pa.list_(pa.float64())),  # Size: n_sigma + 4
    pa.field("knots_rate", pa.list_(pa.float64())),        # Size: n_r + 4

    # ========================================================================
    # B-SPLINE COEFFICIENTS (4D tensor, row-major layout)
    # ========================================================================
    # Size: n_m × n_tau × n_sigma × n_r doubles
    # Index formula: ((i_m * n_tau + i_tau) * n_sigma + i_sigma) * n_r + i_r
    pa.field("coefficients", pa.list_(pa.float64())),

    # ========================================================================
    # OPTIONAL: RAW PRICES (For validation/debugging)
    # ========================================================================
    # Same layout as coefficients, but raw PDE prices (before B-spline fit)
    pa.field("prices_raw", pa.list_(pa.float64()), nullable=True),

    # ========================================================================
    # FITTING STATISTICS (Diagnostic metadata)
    # ========================================================================
    pa.field("max_residual_moneyness", pa.float64()),
    pa.field("max_residual_maturity", pa.float64()),
    pa.field("max_residual_volatility", pa.float64()),
    pa.field("max_residual_rate", pa.float64()),
    pa.field("max_residual_overall", pa.float64()),

    pa.field("condition_number_max", pa.float64()),

    # ========================================================================
    # BUILD METADATA
    # ========================================================================
    pa.field("n_pde_solves", pa.uint32()),           # Total PDE solves performed
    pa.field("precompute_time_seconds", pa.float64()), # Wall-clock build time
    pa.field("pde_n_space", pa.uint32()),            # PDE spatial grid points
    pa.field("pde_n_time", pa.uint32()),             # PDE time steps

    # ========================================================================
    # CHECKSUMS (Data integrity)
    # ========================================================================
    pa.field("checksum_coefficients", pa.uint64()),  # CRC64 of coefficient array
    pa.field("checksum_grids", pa.uint64()),         # CRC64 of all grid vectors
])
```

### File Layout (Arrow IPC Feather V2)

```
┌─────────────────────────────────────────────────────────────┐
│ Arrow IPC Header (Magic: "ARROW1")                          │
├─────────────────────────────────────────────────────────────┤
│ Schema Block (self-describing metadata)                     │
│   - Field names, types, nullability                         │
│   - Custom metadata: {"mango_format_version": "1.0"}        │
├─────────────────────────────────────────────────────────────┤
│ Record Batch 0 (single row with all data)                   │
│   ┌───────────────────────────────────────────────────────┐ │
│   │ Metadata Columns (scalars)                            │ │
│   │   - format_version, timestamp, ticker, etc.           │ │
│   ├───────────────────────────────────────────────────────┤ │
│   │ Grid Vectors (1D arrays, 64-byte aligned)             │ │
│   │   Buffer 1: moneyness [n_m doubles]                   │ │
│   │   Buffer 2: maturity [n_tau doubles]                  │ │
│   │   Buffer 3: volatility [n_sigma doubles]              │ │
│   │   Buffer 4: rate [n_r doubles]                        │ │
│   ├───────────────────────────────────────────────────────┤ │
│   │ Knot Vectors (1D arrays, 64-byte aligned)             │ │
│   │   Buffer 5: knots_m [n_m+4 doubles]                   │ │
│   │   Buffer 6: knots_tau [n_tau+4 doubles]               │ │
│   │   Buffer 7: knots_sigma [n_sigma+4 doubles]           │ │
│   │   Buffer 8: knots_r [n_r+4 doubles]                   │ │
│   ├───────────────────────────────────────────────────────┤ │
│   │ Coefficients (1D array, 64-byte aligned)              │ │
│   │   Buffer 9: coeffs [n_m×n_tau×n_sigma×n_r doubles]    │ │
│   ├───────────────────────────────────────────────────────┤ │
│   │ Optional: Raw Prices (nullable)                       │ │
│   │   Buffer 10: prices_raw [same size as coeffs]         │ │
│   ├───────────────────────────────────────────────────────┤ │
│   │ Statistics & Checksums (scalars)                      │ │
│   └───────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Footer (points back to schema and record batches)           │
└─────────────────────────────────────────────────────────────┘
```

## Memory Layout Specification

### Buffer Alignment

**All numeric buffers MUST be 64-byte aligned** for AVX-512 SIMD:
- Grid vectors: 64-byte boundary
- Knot vectors: 64-byte boundary
- Coefficient tensor: 64-byte boundary

Arrow IPC automatically handles alignment through padding.

### Coefficient Array Layout

**Row-major 4D tensor** (fastest varying index = rate):
```cpp
// Pseudocode index calculation
size_t index(size_t i_m, size_t i_tau, size_t i_sigma, size_t i_r) {
    return ((i_m * n_tau + i_tau) * n_sigma + i_sigma) * n_r + i_r;
}

// Example: 10×8×5×4 grid = 1,600 doubles = 12.8 KB
```

**Cache-friendly access pattern** for evaluation:
```cpp
// eval(m, tau, sigma, r) accesses coefficients with stride-1 in r dimension
for (int i_m = ...) {
    for (int i_tau = ...) {
        for (int i_sigma = ...) {
            // These 4 accesses are sequential in memory:
            c[..., ..., ..., i_r+0]
            c[..., ..., ..., i_r+1]
            c[..., ..., ..., i_r+2]
            c[..., ..., ..., i_r+3]
        }
    }
}
```

## Validation Rules

### Load-Time Validation (Fail-Fast)

```cpp
expected<PriceTableSurface, LoadError> load_price_table(const char* path) {
    // 1. Validate Arrow magic header
    if (!starts_with("ARROW1")) return LoadError::NOT_ARROW_FILE;

    // 2. Validate schema version
    uint32_t version = table["format_version"];
    if (version != 1) return LoadError::UNSUPPORTED_VERSION;

    // 3. Validate dimensions
    size_t n_m = table["n_moneyness"];
    size_t n_tau = table["n_maturity"];
    size_t n_sigma = table["n_volatility"];
    size_t n_r = table["n_rate"];

    if (n_m < 4 || n_tau < 4 || n_sigma < 4 || n_r < 4) {
        return LoadError::INSUFFICIENT_GRID_POINTS;
    }

    // 4. Validate array sizes
    if (table["moneyness"].length() != n_m) return LoadError::SIZE_MISMATCH;
    if (table["knots_moneyness"].length() != n_m + 8) return LoadError::SIZE_MISMATCH;

    size_t expected_coeffs = n_m * n_tau * n_sigma * n_r;
    if (table["coefficients"].length() != expected_coeffs) {
        return LoadError::COEFFICIENT_SIZE_MISMATCH;
    }

    // 5. Validate checksums
    uint64_t checksum = compute_crc64(table["coefficients"].data());
    if (checksum != table["checksum_coefficients"]) {
        return LoadError::CORRUPTED_COEFFICIENTS;
    }

    // 6. Validate grid monotonicity
    if (!is_strictly_increasing(table["moneyness"])) {
        return LoadError::GRID_NOT_SORTED;
    }

    // 7. Build workspace with zero-copy spans
    return build_surface_from_validated_table(table);
}
```

### Error Codes

```cpp
enum class LoadError {
    NOT_ARROW_FILE,              // Missing "ARROW1" magic
    UNSUPPORTED_VERSION,         // format_version != 1
    INSUFFICIENT_GRID_POINTS,    // n < 4 for any axis
    SIZE_MISMATCH,               // Array length doesn't match metadata
    COEFFICIENT_SIZE_MISMATCH,   // coeffs.size() != n_m×n_tau×n_sigma×n_r
    CORRUPTED_COEFFICIENTS,      // CRC64 checksum failed
    CORRUPTED_GRIDS,             // Grid checksum failed
    GRID_NOT_SORTED,             // Monotonicity violation
    MMAP_FAILED,                 // OS mmap error
    INVALID_ALIGNMENT,           // Buffer not 64-byte aligned
};
```

## Schema Versioning Strategy

### Version 1.0 (Initial Release)

**Fields:**
- Core: grids, knots, coefficients, K_ref, dividend_yield
- Optional: prices_raw (nullable)
- Metadata: build stats, checksums

### Future Extensions (Backward Compatible)

**Version 1.1 (Hypothetical):**
- Add `gamma_coefficients` field (nullable) for second derivative
- Old loaders ignore unknown fields (Arrow schema evolution)
- New loaders check field existence before accessing

**Version 2.0 (Breaking Change):**
- Change coefficient layout (e.g., column-major)
- Requires explicit migration tool
- Loader rejects version 2 files immediately

### Compatibility Matrix

| File Version | Loader Version | Result                    |
|--------------|----------------|---------------------------|
| 1.0          | 1.0            | ✅ Full compatibility     |
| 1.1          | 1.0            | ✅ Load (ignore new fields)|
| 1.0          | 1.1            | ✅ Load (missing fields = null)|
| 2.0          | 1.x            | ❌ Reject (LoadError::UNSUPPORTED_VERSION)|

## File Size Analysis

### Example: SPY Put Surface

**Grid dimensions:**
- Moneyness: 50 points (0.7 to 1.3)
- Maturity: 30 points (1 week to 2 years)
- Volatility: 20 points (10% to 50%)
- Rate: 10 points (0% to 10%)

**Data sizes:**
```
Grids:        (50 + 30 + 20 + 10) × 8 bytes = 880 bytes
Knots:        (58 + 38 + 28 + 18) × 8 bytes = 1,136 bytes
Coefficients: (50×30×20×10) × 8 bytes       = 2,400,000 bytes (~2.3 MB)
Metadata:     ~500 bytes
Arrow overhead: ~1 KB

Total: ~2.4 MB per table
```

**For 100 tables:** ~240 MB (fits in L3 cache on modern CPUs)

## Implementation Roadmap

### Phase 1: Schema Definition ✅ (This Document)
- [x] Define Arrow schema
- [x] Specify validation rules
- [x] Document versioning strategy

### Phase 2: Workspace Refactoring (Next)
- [ ] Create `PriceTableWorkspace` with aligned arena
- [ ] Refactor `BSpline4D` to accept workspace reference
- [ ] Add `[[deprecated]]` overloads for compatibility
- [ ] Validate zero performance regression

### Phase 3: Arrow Integration
- [ ] Add Arrow C++ dependency to Bazel
- [ ] Implement `save_price_table()` writer
- [ ] Implement `load_price_table()` mmap reader
- [ ] Add CRC64 checksum validation

### Phase 4: Production Deployment
- [ ] Batch save script for overnight computation
- [ ] Pre-market load script with error handling
- [ ] Monitoring/alerting for corrupted files
- [ ] Migration tool for schema upgrades

## Example Usage

### Save (Overnight Batch)

```cpp
#include "src/option/price_table_4d_builder.hpp"
#include "src/option/price_table_arrow.hpp"

// Build price table (expensive: ~3 minutes)
auto builder = PriceTable4DBuilder::create(grid);
auto result = builder.precompute(config).value();

// Save to Arrow IPC file (<1ms)
auto save_result = save_price_table(result.surface, "SPY_put_20250113.arrow");
if (!save_result) {
    std::cerr << "Save failed: " << save_result.error() << "\n";
}
```

### Load (Pre-Market)

```cpp
#include "src/option/price_table_arrow.hpp"
#include "src/option/iv_solver_interpolated.hpp"

// Load table from disk (<1ms, zero-copy mmap)
auto surface_result = load_price_table("SPY_put_20250113.arrow");
if (!surface_result) {
    std::cerr << "Load failed: " << surface_result.error() << "\n";
    return;
}

PriceTableSurface surface = std::move(surface_result.value());

// Use for IV calculations (135ns per query)
IVSolverInterpolated iv_solver(surface);
auto iv = iv_solver.solve(query);
```

### Batch Load (100 Tables)

```cpp
std::vector<PriceTableSurface> tables;
tables.reserve(100);

// Load all tables in parallel
#pragma omp parallel for
for (const auto& path : table_paths) {
    auto result = load_price_table(path.c_str());
    if (result) {
        #pragma omp critical
        tables.push_back(std::move(result.value()));
    } else {
        std::cerr << "Failed to load " << path << ": " << result.error() << "\n";
    }
}

// Total load time: ~100ms for 100 tables
```

## Performance Targets

| Operation                  | Target       | Notes                          |
|----------------------------|--------------|--------------------------------|
| Save single table          | <10ms        | One-time cost overnight        |
| Load single table (mmap)   | <1ms         | Zero-copy, validation only     |
| Load 100 tables            | <100ms       | Parallel load, 16 cores        |
| Query after load           | ~135ns       | No regression vs in-memory     |
| File size per table        | 2-5 MB       | Dominated by coefficients      |

## Open Questions

1. **Compression:** Should we use Arrow's built-in compression (LZ4, ZSTD)?
   - Pro: Smaller files (~50% reduction)
   - Con: Defeats mmap zero-copy (must decompress)
   - **Decision:** No compression for intraday files, optional for archival

2. **Incremental Updates:** How to handle volatility changes during trading?
   - Option A: Reload entire table (1ms overhead)
   - Option B: Hot-swap coefficients (complex, error-prone)
   - **Recommendation:** Reload entire table (1ms is acceptable)

3. **Multi-Tenor Tables:** Should we pack multiple maturities into one file?
   - Pro: Fewer files to manage
   - Con: Larger mmap footprint, harder to update
   - **Decision:** One table per (ticker, option_type, maturity_bucket)

## References

- Apache Arrow IPC Format: https://arrow.apache.org/docs/format/Columnar.html
- Feather V2 Specification: https://arrow.apache.org/docs/python/feather.html
- B-Spline Implementation: `src/interpolation/bspline_4d.hpp`
- Workspace Pattern: `src/option/american_solver_workspace.hpp`

---

**Next Action:** Implement `PriceTableWorkspace` with aligned arena and refactor `BSpline4D` to accept workspace reference.
