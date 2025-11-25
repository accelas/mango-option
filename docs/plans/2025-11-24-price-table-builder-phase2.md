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
    deps = [...],
)
```

**Reference:** See `american_option_batch` target (line 159-166) which already has OpenMP enabled.

### Code Changes

```cpp
// src/option/price_table_builder.cpp

// Add include
#include <omp.h>

// In extract_tensor():
#pragma omp parallel for collapse(2) schedule(static)
for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
    for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
        // ... extract prices from batch result
    }
}
```

**Expected speedup:** ~10-16× on multi-core machines (only achievable after build system changes)

### Testing
- [ ] Verify `-fopenmp` is passed to compiler (check build log)
- [ ] Verify identical results with `OMP_NUM_THREADS=1` vs default
- [ ] Benchmark extraction time improvement
- [ ] Test with different grid sizes

---

## Improvement 2: Partial Failure Tolerance + Tracking [High Priority]

### Current Behavior (Blocking Issue)

`build()` aborts immediately if `solve_batch` reports any failure:

```cpp
// src/option/price_table_builder.cpp:90-96
auto batch_result = solve_batch(batch_params, axes);
if (batch_result.failed_count > 0) {
    return std::unexpected(
        "solve_batch had " + std::to_string(batch_result.failed_count) +
        " failures out of " + std::to_string(batch_result.results.size()));
}
```

This means `extract_tensor()` is **only ever called with 100% successful batches**. The `if (!result_expected.has_value())` branch inside `extract_tensor()` is dead code.

### Required Upstream Change

Before atomic failure tracking is meaningful, `build()` must tolerate partial failures:

```cpp
// OPTION A: Configurable failure tolerance
struct PriceTableConfig {
    // ... existing fields ...
    double max_failure_rate = 0.0;  // 0.0 = strict (current), 0.1 = allow 10% failures
};

// In build():
auto batch_result = solve_batch(batch_params, axes);
double failure_rate = static_cast<double>(batch_result.failed_count) / batch_result.results.size();
if (failure_rate > config_.max_failure_rate) {
    return std::unexpected("solve_batch failure rate " + std::to_string(failure_rate * 100) +
                           "% exceeds threshold " + std::to_string(config_.max_failure_rate * 100) + "%");
}
// Continue to extract_tensor with partial results...
```

```cpp
// OPTION B: Warning + proceed (always best-effort)
auto batch_result = solve_batch(batch_params, axes);
if (batch_result.failed_count > 0) {
    // Log warning via USDT probe, but continue
    MANGO_TRACE_WARNING(MODULE_PRICE_TABLE, batch_result.failed_count, batch_result.results.size());
}
// Continue to extract_tensor...
```

### Atomic Failure Tracking (After Upstream Change)

Once partial failures can reach `extract_tensor()`:

```cpp
std::atomic<size_t> failed_slices{0};

#pragma omp parallel for collapse(2) schedule(static)
for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
    for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
        if (!result_expected.has_value()) {
            failed_slices.fetch_add(Nt, std::memory_order_relaxed);
            // Fill with NaN
            continue;
        }
        // ...
    }
}
```

### Return Type Change Required

Current return type cannot surface failure count:

```cpp
// CURRENT: src/option/price_table_builder.hpp:205
std::expected<PriceTensor<N>, std::string> extract_tensor(...);
```

**Options:**

```cpp
// OPTION A: Wrap in result struct
struct ExtractionResult {
    PriceTensor<N> tensor;
    size_t failed_slices;
    size_t total_slices;
};
std::expected<ExtractionResult, std::string> extract_tensor(...);

// OPTION B: Add to existing PriceTableResult (preferred - less API churn)
struct PriceTableResult {
    std::shared_ptr<const PriceTableSurface<N>> surface;
    size_t n_pde_solves;
    double build_time_seconds;
    BSplineFittingStats fitting_stats;
    size_t failed_extraction_slices;  // NEW
    size_t total_extraction_slices;   // NEW
};
```

### Implementation Order

1. Change `build()` to tolerate partial failures (add `max_failure_rate` config)
2. Add failure tracking to `extract_tensor()` with atomic counter
3. Extend `PriceTableResult` to include failure stats
4. Add tests for partial failure scenarios

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

## Improvement 5: TimeDomain in Factory Methods [Deferred]

**Problem:** Factory methods take `size_t n_time` which is less expressive.

**Deferred:** Until usage patterns clarify whether users need dt-based or n_steps-based control.

---

## Improvement 6: build_with_save() [Deferred]

**Problem:** No way to stream `PriceTensor<N>` to disk for external analysis.

**Deferred:** Depends on #3 (error types) for proper failure handling.

---

## Implementation Order

| Order | Improvement | Effort | Blockers |
|-------|-------------|--------|----------|
| 1a | Build system: add OpenMP to price_table_builder | Low | None |
| 1b | Parallelize extract_tensor() | Low | 1a |
| 2a | Change build() to tolerate partial failures | Medium | None |
| 2b | Add atomic failure tracking | Low | 2a |
| 2c | Extend PriceTableResult with failure stats | Low | 2b |
| 3 | Enriched error messages (Option C) | Low | None |
| 4 | C++23 ranges for log_moneyness | Low | None |

## Verification

Before completing Phase 2:
- [ ] All 68+ tests pass
- [ ] Benchmarks show expected speedup from parallelization
- [ ] Partial failure tests verify tracking works
- [ ] Examples compile and run correctly
