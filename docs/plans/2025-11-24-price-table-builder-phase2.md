# PriceTableBuilder Phase 2: Performance & Quality Improvements

**Date:** 2025-11-24
**Status:** Planning
**Prerequisite:** Phase 1 complete (PR #245 merged)

## Overview

Phase 1 delivered a generic `PriceTableBuilder<N>` with factory methods and custom grid support. Phase 2 focuses on performance optimizations and API quality improvements.

## Improvements

### 1. Parallelize extract_tensor() [High Priority]

**Problem:** `extract_tensor()` loops over (σ,r) batches serially.

**Location:** `src/option/price_table_builder.cpp:339-389`

**Impact:** Leaves cores idle during extraction. For 20×10 = 200 batches, this is embarrassingly parallel work.

**Solution:** Add OpenMP to outer loop:

```cpp
// BEFORE (serial):
for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
    for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
        // ... extract prices from batch result
    }
}

// AFTER (parallel):
#pragma omp parallel for collapse(2) schedule(static)
for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
    for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
        // ... extract prices from batch result
    }
}
```

**Expected speedup:** ~10-16× on multi-core machines

**Testing:**
- Verify identical results with/without parallelization
- Benchmark extraction time improvement
- Test with different grid sizes

---

### 2. Atomic Failure Tracking [High Priority]

**Problem:** `extract_tensor()` fills NaN for failures but doesn't count them.

**Impact:** No way to detect partial failures or get useful error messages.

**Solution:** Add atomic counter for thread-safe failure tracking:

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

        for (size_t j = 0; j < Nt; ++j) {
            if (build_error.has_value()) {
                failed_slices.fetch_add(1, std::memory_order_relaxed);
                // Fill with NaN
                continue;
            }
        }
    }
}

// Return failure count in result or error if threshold exceeded
```

**Dependency:** Should be implemented together with #1 (parallelization).

---

### 3. Structured Error Types [Medium Priority]

**Problem:** All methods return `std::string` errors - no programmatic error handling.

**Solution:** Add `PriceTableError` enum and struct:

```cpp
enum class PriceTableErrorCode {
    // Validation errors
    INVALID_GRID_UNSORTED,
    INVALID_GRID_NEGATIVE,
    INVALID_GRID_EMPTY,
    INVALID_GRID_TOO_FEW_POINTS,
    INVALID_K_REF,
    INVALID_DOMAIN_COVERAGE,

    // Build errors
    INCOMPLETE_RESULTS,        // Some PDE solves failed
    BSPLINE_FIT_FAILED,

    // Runtime errors
    UNSUPPORTED_DIMENSION,
    IO_ERROR
};

struct PriceTableError {
    PriceTableErrorCode code;
    std::string message;                    // Human-readable description
    std::optional<size_t> axis_index;       // 0=m, 1=τ, 2=σ, 3=r
    std::optional<double> invalid_value;    // The problematic value
    std::optional<size_t> failed_count;     // For INCOMPLETE_RESULTS
};
```

**API change:**
```cpp
// Before
std::expected<PriceTableResult<N>, std::string> build(...);

// After
std::expected<PriceTableResult<N>, PriceTableError> build(...);
```

**Migration:** Provide `error.message` for backwards-compatible string access.

---

### 4. C++23 Ranges in extract_tensor() [Low Priority]

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

### 5. TimeDomain in Factory Methods [Low Priority - Deferred]

**Problem:** Factory methods take `size_t n_time` which is less expressive.

**Current API:**
```cpp
from_vectors(..., GridSpec<double> grid_spec, size_t n_time, OptionType type, ...)
```

**Options:**

| Option | API | Pros | Cons |
|--------|-----|------|------|
| A | `TimeDomain time` | Reuses existing type | t_end unknown until axes |
| B | `TimeConfig{n_steps}` | Clear intent | New type |
| C | `std::variant<size_t, double>` | n_steps or dt | Implicit |

**Recommendation:** Defer until usage patterns clarify requirements.

---

### 6. build_with_save() [Low Priority - Deferred]

**Problem:** No way to stream `PriceTensor<N>` to disk for external analysis.

**Proposed API:**
```cpp
std::expected<PriceTableResult<N>, PriceTableError>
build_with_save(
    const PriceTableAxes<N>& axes,
    const std::filesystem::path& output_path);
```

**Atomicity contract:**
- Write to `{output_path}.tmp` during extraction
- Atomic rename to `{output_path}` on success
- Delete `.tmp` on failure
- Clean stale `.tmp` files on startup (crash recovery)

**Use case:** External tools need raw PDE prices for validation/analysis.

---

## Implementation Order

| Order | Improvement | Effort | Dependencies |
|-------|-------------|--------|--------------|
| 1 | Parallelize extraction (#1) | Low | None |
| 2 | Atomic failure tracking (#2) | Low | #1 |
| 3 | Structured error types (#3) | Medium | None |
| 4 | C++23 ranges (#4) | Low | None |
| 5 | TimeDomain factories (#5) | Low | Deferred |
| 6 | build_with_save (#6) | Medium | #3 |

## Verification

Before completing Phase 2:
- [ ] All 68+ tests pass
- [ ] Benchmarks show expected speedup from parallelization
- [ ] Examples compile and run correctly
- [ ] Documentation updated for API changes
