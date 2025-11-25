# PriceTableBuilder Phase 2: Performance & Quality Improvements

**Date:** 2025-11-24
**Status:** Planning
**Prerequisite:** Phase 1 complete (PR #245 merged)

## Overview

Phase 1 delivered a generic `PriceTableBuilder<N>` with factory methods and custom grid support. Phase 2 focuses on performance optimizations and API quality improvements.

---

## Improvement 1: Parallelize extract_tensor() [High Priority]

**Problem:** `extract_tensor()` loops over (σ,r) batches serially.

**Location:** `src/option/price_table_builder.cpp:339-389`

**Impact:** Leaves cores idle during extraction. For 20×10 = 200 batches, this is embarrassingly parallel work.

### Build System Changes Required

The `price_table_builder` Bazel target currently has no OpenMP flags:

```python
# src/option/BUILD.bazel:55-74 (CURRENT - no OpenMP)
cc_library(
    name = "price_table_builder",
    srcs = ["price_table_builder.cpp"],
    hdrs = ["price_table_builder.hpp"],
    deps = [...],
)
```

**Required change:**

```python
# src/option/BUILD.bazel (AFTER)
cc_library(
    name = "price_table_builder",
    srcs = ["price_table_builder.cpp"],
    hdrs = ["price_table_builder.hpp"],
    copts = [
        "-fopenmp",
    ],
    linkopts = ["-fopenmp"],
    deps = [
        # ... existing deps ...
        "//src/support:parallel",  # NEW: for MANGO_PRAGMA_* macros
    ],
)
```

**Reference:** See `american_option_batch` target (line 159-166) which already has OpenMP enabled.

### Code Changes

This codebase uses abstraction macros from `src/support/parallel.hpp` for portability across OpenMP/SYCL/sequential backends. Do NOT use raw `#pragma omp` directives.

```cpp
// src/option/price_table_builder.cpp

// Add include (NOT <omp.h> directly)
#include "src/support/parallel.hpp"

// In extract_tensor():
// MANGO_PRAGMA_PARALLEL requires a compound statement (braces)
// See src/option/american_option_batch.cpp:454-500 for reference
MANGO_PRAGMA_PARALLEL
{
    MANGO_PRAGMA_FOR_COLLAPSE2
    for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
        for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
            // ... extract prices from batch result
        }
    }
}
```

**Available macros** (from `src/support/parallel.hpp`):
- `MANGO_PRAGMA_PARALLEL` → `#pragma omp parallel`
- `MANGO_PRAGMA_FOR_COLLAPSE2` → `#pragma omp for collapse(2)`

**Expected speedup:** ~10-16× on multi-core machines (only achievable after build system changes)

### Testing
- [ ] Verify `-fopenmp` is passed to compiler (check build log)
- [ ] Verify identical results with `OMP_NUM_THREADS=1` vs default
- [ ] Benchmark extraction time improvement
- [ ] Test with different grid sizes

---

## Improvement 2: Partial Failure Tolerance + Tracking [High Priority]

### Problem Summary

Three blocking issues prevent partial failure tolerance:

1. **`build()` aborts on any failure** (src/option/price_table_builder.cpp:90-96)
2. **`extract_tensor()` fills NaN for failed slices** (src/option/price_table_builder.cpp:331-369)
3. **`BSplineNDSeparable::fit()` rejects NaN values** (src/math/bspline_nd_separable.hpp:152-154)

Even if we relax `build()` to tolerate partial failures, the fitting step will fail:

```cpp
// src/math/bspline_nd_separable.hpp:149-154
for (size_t i = 0; i < values.size(); ++i) {
    if (std::isnan(values[i])) {
        return std::unexpected(
            "Input values contain NaN at index " + std::to_string(i));
    }
}
```

### Required Changes (In Order)

#### Step 1: Change `build()` to tolerate partial PDE failures

```cpp
struct PriceTableConfig {
    // ... existing fields ...
    double max_failure_rate = 0.0;  // 0.0 = strict (current), 0.1 = allow 10%
};

// In build():
auto batch_result = solve_batch(batch_params, axes);
double failure_rate = static_cast<double>(batch_result.failed_count) / batch_result.results.size();
if (failure_rate > config_.max_failure_rate) {
    return std::unexpected("solve_batch failure rate exceeds threshold");
}
// Continue to extract_tensor with partial results...
```

#### Step 2: Change `extract_tensor()` return type to surface failure counts

```cpp
// NEW: Extraction result with failure tracking
template <size_t N>
struct ExtractionResult {
    PriceTensor<N> tensor;
    size_t total_slices;
    std::vector<size_t> failed_indices;  // Flat indices of failed slices (sparse)
};

// CHANGED: extract_tensor returns ExtractionResult
template <size_t N>
std::expected<ExtractionResult<N>, std::string>
PriceTableBuilder<N>::extract_tensor(
    const BatchAmericanOptionResult& batch,
    const PriceTableAxes<N>& axes) const;

// In build(), plumb the counts through:
auto extraction = extract_tensor(batch_result, axes);
if (!extraction.has_value()) {
    return std::unexpected("extract_tensor failed: " + extraction.error());
}
size_t failed_extraction_slices = extraction->failed_indices.size();
size_t total_extraction_slices = extraction->total_slices;
// ... continue to fit_coeffs with extraction->tensor ...
```

**Collecting failed indices in parallel loop** (inside `extract_tensor()`):
```cpp
// Thread-safe collection of failed indices
std::vector<size_t> failed_indices;
std::mutex failed_mutex;  // For thread-safe push_back

// MANGO_PRAGMA_PARALLEL requires a compound statement (braces)
// See src/option/american_option_batch.cpp:454-500 for reference
MANGO_PRAGMA_PARALLEL
{
    MANGO_PRAGMA_FOR_COLLAPSE2
    for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
        for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
            size_t batch_idx = σ_idx * Nr + r_idx;
            if (!batch.results[batch_idx].has_value()) {
                {
                    std::lock_guard<std::mutex> lock(failed_mutex);
                    failed_indices.push_back(batch_idx);
                }
                // ... fill with NaN ...
            }
            // ...
        }
    }
}
// failed_indices now contains all failed (σ,r) flat indices
```

Note: Since failures are rare, the mutex contention is negligible.

#### Step 3: Handle NaN values before fitting

**Option A: Pre-fill failed slices with neighbor interpolation**
```cpp
// After extraction, before fitting (in build()):
// Iterate only over failed indices (sparse, efficient)
auto& extraction_result = extraction.value();
for (size_t flat_idx : extraction_result.failed_indices) {
    size_t σ_idx = flat_idx / Nr;
    size_t r_idx = flat_idx % Nr;
    interpolate_from_neighbors(extraction_result.tensor, σ_idx, r_idx, Nσ, Nr);
}
// Now tensor has no NaN, safe to call fit_coeffs()
```

**Option B: Modify fitter to skip NaN slices (complex)**
```cpp
// In BSplineNDSeparable::fit():
// Skip 1D fits that contain NaN, mark as failed
// Requires significant fitter redesign
```

**Option C: Replace NaN with boundary extrapolation**
```cpp
// For failed interior slices, use boundary values
// Simple but may introduce artifacts
```

**Recommendation:** Option A (neighbor interpolation) provides best quality while being implementable without fitter changes.

#### Step 4: Extend `PriceTableResult` with failure stats

```cpp
template <size_t N>
struct PriceTableResult {
    std::shared_ptr<const PriceTableSurface<N>> surface;
    size_t n_pde_solves;
    double build_time_seconds;
    BSplineFittingStats fitting_stats;
    // NEW: Failure tracking
    size_t failed_pde_solves;           // From batch_result.failed_count
    size_t failed_extraction_slices;    // From extraction.failed_slices
    size_t total_extraction_slices;     // From extraction.total_slices
};
```

### Data Flow Summary

```
solve_batch() → batch_result.failed_count
                      ↓
extract_tensor() → ExtractionResult{tensor, total_slices, failed_indices}
                      ↓
interpolate_failed_slices(failed_indices) → tensor with NaN replaced
                      ↓
fit_coeffs() → FitCoeffsResult (no NaN, fitting succeeds)
                      ↓
build() → PriceTableResult{..., failed_pde_solves, failed_extraction_slices, ...}
```

### Implementation Order

1. Change `extract_tensor()` return type to `ExtractionResult<N>`
2. Add atomic failure counting in `extract_tensor()` (parallel-safe)
3. Add `interpolate_failed_slices()` helper to replace NaN with interpolated values
4. Change `build()` to accept partial failures (add `max_failure_rate` config)
5. Extend `PriceTableResult` with failure stats
6. Add tests for partial failure scenarios

---

## Improvement 3: Integrate with Existing Error Types [Medium Priority]

### Problem with Original Plan

The original plan proposed creating a new `PriceTableError` hierarchy, but this project already has structured error types in `src/support/error_types.hpp`:

- `ValidationError` with `ValidationErrorCode` (InvalidStrike, InvalidMaturity, UnsortedGrid, etc.)
- `SolverError` with `SolverErrorCode` (Stage1ConvergenceFailure, etc.)
- `AllocationError` with `AllocationErrorCode`
- `InterpolationError` with `InterpolationErrorCode`

### Conversion Strategy

`PriceTableBuilder` methods return `std::expected<T, std::string>`. To integrate with existing error types:

**Option A: Use std::variant for multiple error types**
```cpp
using PriceTableError = std::variant<
    ValidationError,
    SolverError,
    AllocationError,
    InterpolationError,
    std::string  // Fallback for unstructured errors
>;

std::expected<PriceTableResult<N>, PriceTableError> build(...);
```

**Option B: Add PriceTableErrorCode to existing system**
```cpp
// In src/support/error_types.hpp, add:
enum class PriceTableErrorCode {
    InvalidDomainCoverage,
    PartialPDEFailure,
    ExtractionFailure,
    FittingFailure
};

struct PriceTableError {
    PriceTableErrorCode code;
    std::optional<ValidationError> validation_error;  // If validation caused it
    std::optional<SolverError> solver_error;          // If solver caused it
    std::string message;
};
```

**Option C: Keep std::string but enrich messages (minimal change)**
```cpp
// Keep current signature
std::expected<PriceTableResult<N>, std::string> build(...);

// But ensure error messages include structured info:
return std::unexpected(
    "Validation failed: " + validation_error_to_string(err) +
    " [code=" + std::to_string(static_cast<int>(err.code)) +
    ", value=" + std::to_string(err.value) + "]");
```

### Recommendation

Start with **Option C** (minimal change, enriched strings) for Phase 2. Consider **Option B** for a future Phase 3 if programmatic error handling becomes important.

---

## Improvement 4: C++23 Ranges in extract_tensor() [Low Priority]

**Problem:** Explicit loops could be modernized.

**Current code:**
```cpp
std::vector<double> log_moneyness(Nm);
for (size_t i = 0; i < Nm; ++i) {
    log_moneyness[i] = std::log(axes.grids[0][i]);
}
```

**Modernized:**
```cpp
auto log_moneyness = axes.grids[0]
    | std::views::transform([](double m) { return std::log(m); })
    | std::ranges::to<std::vector>();
```

**Scope:**
- **Do now:** `std::views::transform` for log_moneyness (clear win)
- **Defer:** `cartesian_product` - nested loops are clearer for error handling
- **Defer:** `submdspan` - limited library support

**Dependencies:** C++23 `<ranges>`, `std::ranges::to`

---

## Improvement 5: Unified Grid Config API [Medium Priority]

**Problem:** Factory methods take `GridSpec` and `size_t n_time` as separate required parameters, inconsistent with batch solver pattern.

**Current API:**
```cpp
// PriceTableBuilder factories (current)
from_vectors(..., GridSpec<double> grid_spec, size_t n_time, OptionType type, ...);

// BatchAmericanOptionSolver (existing pattern)
solve_batch(..., std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid = std::nullopt);
```

**Proposed unified API:**

```cpp
using GridConfig = std::pair<GridSpec<double>, TimeDomain>;

// Factory methods with optional grid config
static std::expected<std::pair<PriceTableBuilder<N>, PriceTableAxes<N>>, std::string>
from_vectors(
    std::vector<double> moneyness,
    std::vector<double> maturity,
    std::vector<double> volatility,
    std::vector<double> rate,
    double K_ref,
    std::optional<GridConfig> grid_config = std::nullopt,  // NEW: unified
    OptionType type = OptionType::PUT,
    double dividend_yield = 0.0);

// Usage:
// Auto-estimation (default):
auto [builder, axes] = PriceTableBuilder<4>::from_vectors(m, tau, sigma, r, K_ref).value();

// Custom grid:
auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0).value();
auto time_domain = TimeDomain::from_n_steps(0.0, 2.0, 1000);
auto [builder, axes] = PriceTableBuilder<4>::from_vectors(
    m, tau, sigma, r, K_ref,
    std::make_pair(grid_spec, time_domain)
).value();
```

**Benefits:**
- Consistent with `BatchAmericanOptionSolver::solve_batch()` API
- `TimeDomain` provides both `n_steps` and `dt` access
- Auto-estimation when grid_config is nullopt
- Single optional parameter instead of two required ones

**Implementation:**
1. Add `using GridConfig = std::pair<GridSpec<double>, TimeDomain>;` to header
2. Change factory signatures to use `std::optional<GridConfig>`
3. When nullopt: estimate grid from max maturity and volatility range
4. When provided: validate TimeDomain.t_end() matches max maturity (or use it)
5. Store in `PriceTableConfig` as optional

---

## Improvement 6: build_with_save() [Deferred]

**Problem:** No way to stream `PriceTensor<N>` to disk for external analysis.

**Proposed API (using unified GridConfig from #5):**

```cpp
std::expected<PriceTableResult<N>, std::string>
build_with_save(
    const PriceTableAxes<N>& axes,
    const std::filesystem::path& output_path);
```

**Atomicity contract:**
- Write to `{output_path}.tmp` during extraction
- Atomic rename to `{output_path}` on success
- Delete `.tmp` on failure
- Clean stale `.tmp` files on startup (crash recovery)

**Deferred:** Depends on #3 (error types) for proper failure handling.

---

## Implementation Order

| Order | Improvement | Effort | Blockers |
|-------|-------------|--------|----------|
| 1a | Build system: add OpenMP to price_table_builder | Low | None |
| 1b | Parallelize extract_tensor() | Low | 1a |
| 2a | Change extract_tensor() return type to ExtractionResult | Low | None |
| 2b | Add atomic failure counting in extract_tensor() | Low | 2a, 1b |
| 2c | Add interpolate_failed_slices() helper | Medium | 2a |
| 2d | Change build() to accept partial failures | Low | 2a, 2c |
| 2e | Extend PriceTableResult with failure stats | Low | 2d |
| 3 | Enriched error messages (Option C) | Low | None |
| 4 | C++23 ranges for log_moneyness | Low | None |
| 5 | Unified GridConfig API | Medium | None |
| 6 | build_with_save() | Medium | #3, #5 |

## Verification

Before completing Phase 2:
- [ ] All 68+ tests pass
- [ ] Benchmarks show expected speedup from parallelization
- [ ] Partial failure tests verify tracking works
- [ ] Examples compile and run correctly
