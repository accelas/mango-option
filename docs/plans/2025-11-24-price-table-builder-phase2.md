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
        "-pthread",  # Required for std::mutex
    ],
    linkopts = [
        "-fopenmp",
        "-pthread",  # Required for std::mutex (pthread_mutex_* symbols)
    ],
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

// Add includes (NOT <omp.h> directly)
#include "src/support/parallel.hpp"
#include <mutex>           // std::mutex, std::lock_guard
#include <tuple>           // std::tuple
#include <map>             // std::map
#include <unordered_set>   // std::unordered_set (for early bailout dedup)
#include <vector>          // std::vector (explicit, don't rely on transitive)
#include <optional>        // std::optional (explicit, don't rely on transitive)

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

#### Step 1: Extend `PriceTableConfig` and change `build()` to tolerate partial failures

**In `src/option/price_table_config.hpp`:**
```cpp
// Current struct (src/option/price_table_config.hpp:12-18)
struct PriceTableConfig {
    OptionType option_type = OptionType::PUT;
    double K_ref = 100.0;
    GridSpec<double> grid_estimator;
    size_t n_time = 1000;
    double dividend_yield = 0.0;
    std::vector<std::pair<double, double>> discrete_dividends;  // (time, amount)
    // NEW: Partial failure tolerance
    double max_failure_rate = 0.0;  // 0.0 = strict (current), 0.1 = allow 10%
};

// Validation helper (call in factory methods or build())
// Returns error string on failure, nullopt on success (fits std::expected pipeline)
inline std::optional<std::string> validate_config(const PriceTableConfig& config) {
    if (config.max_failure_rate < 0.0 || config.max_failure_rate > 1.0) {
        return "max_failure_rate must be in [0.0, 1.0], got " +
               std::to_string(config.max_failure_rate);
    }
    return std::nullopt;  // Valid
}

// Usage in build():
// if (auto err = validate_config(config_); err.has_value()) {
//     return std::unexpected(err.value());
// }
```

**Update factory methods** to accept optional `max_failure_rate` and validate:
```cpp
// In from_vectors(), from_strikes(), from_chain():
// Add parameter: double max_failure_rate = 0.0

// Example: from_vectors()
template <size_t N>
static std::expected<PriceTableBuilder<N>, std::string> from_vectors(
    /* existing params */,
    double max_failure_rate = 0.0)
{
    PriceTableConfig config;
    // ... populate config fields ...
    config.max_failure_rate = max_failure_rate;

    // Validate immediately - reject bad configs before any work
    if (auto err = validate_config(config); err.has_value()) {
        return std::unexpected(err.value());
    }

    return PriceTableBuilder<N>(std::move(config), /* axes */);
}

// Same pattern for from_strikes() and from_chain()
```

**In PriceTableBuilder constructor** (defense in depth for direct construction):
```cpp
// Add member variable to store deferred validation error:
// std::string config_validation_error_;  // Empty if valid

template <size_t N>
PriceTableBuilder<N>::PriceTableBuilder(PriceTableConfig config, PriceTableAxes<N> axes)
    : config_(std::move(config)), axes_(std::move(axes))
{
    // Validate even if caller bypasses factory
    if (auto err = validate_config(config_); err.has_value()) {
        // Constructor can't return std::expected, so store error for build() to check
        // Alternative: use static factory pattern exclusively (make constructor private)
        config_validation_error_ = err.value();
    }
}
```

**In `build()`:**
```cpp
auto batch_result = solve_batch(batch_params, axes);
double failure_rate = static_cast<double>(batch_result.failed_count) / batch_result.results.size();
if (failure_rate > config_.max_failure_rate) {
    return std::unexpected("solve_batch failure rate exceeds threshold");
}

// IMPORTANT: Even if failure_rate passes, repair requires at least one valid donor.
// If failure_rate == 1.0 (all failed), repair will fail later with "no valid neighbors".
// Consider enforcing: failure_rate < 1.0 (at least one success required)
if (batch_result.failed_count == batch_result.results.size()) {
    return std::unexpected("All PDE solves failed - no valid donor for repair");
}
// Continue to extract_tensor with partial results...
```

#### Step 2: Change `extract_tensor()` return type to surface failure counts

```cpp
// NEW: Extraction result with failure tracking
// Two failure modes exist in extract_tensor():
// 1. Full PDE failure: !batch.results[batch_idx].has_value() → entire (σ,r) slice NaN
// 2. Per-maturity spline failure: spline.build() fails → single (σ,r,τ) slice NaN
//    (src/option/price_table_builder.cpp:366-371)

template <size_t N>
struct ExtractionResult {
    PriceTensor<N> tensor;
    size_t total_slices;              // Nσ × Nr
    std::vector<size_t> failed_pde;   // Flat (σ,r) indices where PDE failed
    std::vector<std::tuple<size_t, size_t, size_t>> failed_spline;  // (σ,r,τ) tuples
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
size_t failed_pde_slices = extraction->failed_pde.size();
size_t failed_spline_slices = extraction->failed_spline.size();
size_t total_extraction_slices = extraction->total_slices;
// ... continue to fit_coeffs with extraction->tensor ...
```

**Collecting failed indices in parallel loop** (inside `extract_tensor()`):
```cpp
// Thread-safe collection of both failure types
std::vector<size_t> failed_pde;
std::vector<std::tuple<size_t, size_t, size_t>> failed_spline;
std::mutex failed_mutex;  // For thread-safe push_back

MANGO_PRAGMA_PARALLEL
{
    MANGO_PRAGMA_FOR_COLLAPSE2
    for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
        for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
            size_t batch_idx = σ_idx * Nr + r_idx;

            // Failure mode 1: Full PDE solve failed
            if (!batch.results[batch_idx].has_value()) {
                {
                    std::lock_guard<std::mutex> lock(failed_mutex);
                    failed_pde.push_back(batch_idx);
                }
                // Fill entire (m,τ) slice with NaN...
                continue;
            }

            // PDE succeeded, now extract per-maturity
            const auto& result = batch.results[batch_idx].value();
            for (size_t τ_idx = 0; τ_idx < Nt; ++τ_idx) {
                CubicSpline<double> spline;
                auto err = spline.build(x_grid, result.at_time(τ_idx));

                // Failure mode 2: Spline construction failed for this maturity
                if (err.has_value()) {
                    {
                        std::lock_guard<std::mutex> lock(failed_mutex);
                        failed_spline.emplace_back(σ_idx, r_idx, τ_idx);
                    }
                    // Fill (m,:) at this (σ,r,τ) with NaN...
                    continue;
                }
                // Evaluate spline...
            }
        }
    }
}
```

Note: Since failures are rare, mutex contention is negligible.

#### Step 3: Handle NaN values before fitting

The interpolation step MUST guarantee NaN-free output, or abort gracefully. This requires handling both failure modes and edge cases.

**Implementation: `repair_failed_slices()`**

**CRITICAL**: Repair order matters! Partial spline failures must be repaired FIRST
(via τ-interpolation), so that when we do full-slice neighbor copies, the donors
are already NaN-free. If we copy first, we may spread NaN from donors that have
unreported spline failures.

```cpp
// Return type: repair statistics for PriceTableResult
struct RepairStats {
    size_t repaired_full_slices;    // (σ,r) slices: PDE failures + all-maturity spline failures
    size_t repaired_partial_points; // (σ,r,τ) points: Partial spline failures (τ-interpolated)
};

// Returns repair stats on success, error string on failure
std::expected<RepairStats, std::string> repair_failed_slices(
    PriceTensor<N>& tensor,
    const std::vector<size_t>& failed_pde,
    const std::vector<std::tuple<size_t, size_t, size_t>>& failed_spline,
    const PriceTableAxes<N>& axes)
{
    const size_t Nm = axes.grids[0].size();
    const size_t Nt = axes.grids[1].size();
    const size_t Nσ = axes.grids[2].size();
    const size_t Nr = axes.grids[3].size();

    // Group spline failures by (σ,r) to detect full-slice vs partial failures
    std::map<std::pair<size_t, size_t>, std::vector<size_t>> spline_failures_by_slice;
    for (auto [σ_idx, r_idx, τ_idx] : failed_spline) {
        spline_failures_by_slice[{σ_idx, r_idx}].push_back(τ_idx);
    }

    // Collect slices that need full neighbor copy (PDE failures + all-maturity spline failures)
    // Track counts separately for informative error messages
    // IMPORTANT: Deduplicate - a slice can be in both failed_pde AND have all-maturity spline failure
    std::unordered_set<size_t> full_slice_set(failed_pde.begin(), failed_pde.end());
    const size_t n_pde_failures = failed_pde.size();
    size_t n_all_maturity_spline_failures = 0;
    size_t partial_spline_points = 0;  // Count of (σ,r,τ) points, not slices
    for (auto& [slice_key, τ_failures] : spline_failures_by_slice) {
        if (τ_failures.size() == Nt) {
            auto [σ_idx, r_idx] = slice_key;
            size_t flat_idx = σ_idx * Nr + r_idx;
            // Only count as spline failure if not already a PDE failure
            if (full_slice_set.find(flat_idx) == full_slice_set.end()) {
                full_slice_set.insert(flat_idx);
                ++n_all_maturity_spline_failures;
            }
            // (If already in set from PDE failure, it's already counted there)
        } else {
            partial_spline_points += τ_failures.size();
        }
    }
    // Convert to vector for iteration (order doesn't matter for repair)
    std::vector<size_t> full_slice_failures(full_slice_set.begin(), full_slice_set.end());

    // Track which (σ,r) slices are valid donors (no full failures, no partial spline NaN)
    std::vector<bool> slice_valid(Nσ * Nr, true);
    for (size_t flat_idx : full_slice_failures) {
        slice_valid[flat_idx] = false;
    }

    // ========== PHASE 1: Repair partial spline failures via τ-interpolation ==========
    // This makes potential donor slices NaN-free before we copy from them
    for (auto& [slice_key, τ_failures] : spline_failures_by_slice) {
        auto [σ_idx, r_idx] = slice_key;

        // Skip full-slice failures (handled in Phase 2)
        if (τ_failures.size() == Nt) continue;

        // Partial failures: interpolate along τ axis
        for (size_t τ_idx : τ_failures) {
            std::optional<size_t> τ_before, τ_after;
            for (size_t j = τ_idx; j-- > 0; ) {
                if (!std::isnan(tensor.view[0, j, σ_idx, r_idx])) {
                    τ_before = j; break;
                }
            }
            for (size_t j = τ_idx + 1; j < Nt; ++j) {
                if (!std::isnan(tensor.view[0, j, σ_idx, r_idx])) {
                    τ_after = j; break;
                }
            }

            // At least one must exist (not all_maturities_failed)
            for (size_t i = 0; i < Nm; ++i) {
                if (τ_before && τ_after) {
                    double t = static_cast<double>(τ_idx - *τ_before) /
                               static_cast<double>(*τ_after - *τ_before);
                    tensor.view[i, τ_idx, σ_idx, r_idx] =
                        (1.0 - t) * tensor.view[i, *τ_before, σ_idx, r_idx] +
                        t * tensor.view[i, *τ_after, σ_idx, r_idx];
                } else if (τ_before) {
                    tensor.view[i, τ_idx, σ_idx, r_idx] = tensor.view[i, *τ_before, σ_idx, r_idx];
                } else {
                    tensor.view[i, τ_idx, σ_idx, r_idx] = tensor.view[i, *τ_after, σ_idx, r_idx];
                }
            }
        }
    }
    // After Phase 1: all slice_valid slices are now NaN-free

    // ========== PHASE 2: Repair full-slice failures via neighbor copy ==========
    // Now donors are guaranteed NaN-free (Phase 1 cleaned them)
    size_t repaired_full_count = 0;  // Count actual successful repairs
    for (size_t flat_idx : full_slice_failures) {
        size_t σ_idx = flat_idx / Nr;
        size_t r_idx = flat_idx % Nr;

        auto neighbor = find_nearest_valid_neighbor(σ_idx, r_idx, Nσ, Nr, slice_valid);
        if (!neighbor.has_value()) {
            return std::unexpected(
                "Repair failed at slice (" + std::to_string(σ_idx) + "," +
                std::to_string(r_idx) + "): no valid donor. Total full-slice failures: " +
                std::to_string(full_slice_failures.size()) + " (" +
                std::to_string(n_pde_failures) + " PDE + " +
                std::to_string(n_all_maturity_spline_failures) + " all-maturity spline)");
        }
        auto [nσ, nr] = neighbor.value();

        // Copy entire (m,τ) surface from neighbor (now guaranteed NaN-free)
        for (size_t i = 0; i < Nm; ++i) {
            for (size_t j = 0; j < Nt; ++j) {
                tensor.view[i, j, σ_idx, r_idx] = tensor.view[i, j, nσ, nr];
            }
        }

        // Mark as valid so this slice can be a donor for subsequent repairs
        slice_valid[flat_idx] = true;
        ++repaired_full_count;  // Increment only after successful copy
    }

    return RepairStats{
        .repaired_full_slices = repaired_full_count,  // Actual successes, not attempts
        .repaired_partial_points = partial_spline_points
    };
}

// Helper: find nearest valid (σ,r) using Manhattan distance
std::optional<std::pair<size_t, size_t>> find_nearest_valid_neighbor(
    size_t σ_idx, size_t r_idx, size_t Nσ, size_t Nr,
    const std::vector<bool>& slice_valid)
{
    // Max Manhattan distance on grid is (Nσ-1) + (Nr-1)
    const size_t max_dist = (Nσ - 1) + (Nr - 1);

    // Search in expanding rings (Manhattan distance 1, 2, 3, ...)
    for (size_t dist = 1; dist <= max_dist; ++dist) {
        for (int dσ = -static_cast<int>(dist); dσ <= static_cast<int>(dist); ++dσ) {
            int dr = static_cast<int>(dist) - std::abs(dσ);
            for (int sign : {-1, 1}) {
                int nσ = static_cast<int>(σ_idx) + dσ;
                int nr = static_cast<int>(r_idx) + sign * dr;
                if (nσ >= 0 && nσ < static_cast<int>(Nσ) &&
                    nr >= 0 && nr < static_cast<int>(Nr)) {
                    if (slice_valid[nσ * Nr + nr]) {
                        return std::make_pair(static_cast<size_t>(nσ),
                                              static_cast<size_t>(nr));
                    }
                }
            }
        }
    }
    return std::nullopt;  // Entire grid is invalid
}
```

**Guarantees:**
- If `repair_failed_slices()` returns success, tensor contains no NaN
- If repair is impossible (entire axis failed), returns error before `fit_coeffs()` runs
- Caller can decide: abort build, or proceed with degraded accuracy warning

#### Step 4: Extend `PriceTableResult` with failure stats

```cpp
template <size_t N>
struct PriceTableResult {
    std::shared_ptr<const PriceTableSurface<N>> surface;
    size_t n_pde_solves;
    double build_time_seconds;
    BSplineFittingStats fitting_stats;
    // NEW: Failure and repair tracking
    // Note: "slices" = (σ,r) pairs, "points" = (σ,r,τ) triples
    size_t failed_pde_slices;            // Full (σ,r) PDE failures
    size_t failed_spline_points;         // Per-maturity (σ,r,τ) spline failures
    size_t repaired_full_slices;         // (σ,r) slices repaired via neighbor copy
    size_t repaired_partial_points;      // (σ,r,τ) points repaired via τ-interpolation
    size_t total_slices;                 // Nσ × Nr (for slice ratios)
    size_t total_points;                 // Nσ × Nr × Nt (for point ratios)
};
```

**In `build()`, validate and populate:**
```cpp
// Check for config validation errors (if constructor was used directly)
if (!config_validation_error_.empty()) {
    return std::unexpected("Invalid config: " + config_validation_error_);
}

// ... solve_batch, extract_tensor ...

// repair_failed_slices handles "no valid donor" detection internally.
// It uses the same deduplication logic (unordered_set) to count failures
// and returns a detailed error if repair is impossible:
//   "Repair failed at slice (σ,r): no valid donor. Total full-slice failures: N (M PDE + P all-maturity spline)"
// No separate guard needed here - that would duplicate the logic and risk inconsistency.

auto repair_result = repair_failed_slices(extraction.tensor, extraction.failed_pde,
                                          extraction.failed_spline, axes);
if (!repair_result.has_value()) {
    return std::unexpected(repair_result.error());
}
auto repair_stats = repair_result.value();

// ... fit_coeffs() ...

// Populate PriceTableResult:
const size_t Nt = axes.grids[1].size();
return PriceTableResult<N>{
    .surface = ...,
    .n_pde_solves = ...,
    // ...
    .failed_pde_slices = extraction.failed_pde.size(),
    .failed_spline_points = extraction.failed_spline.size(),  // (σ,r,τ) count
    .repaired_full_slices = repair_stats.repaired_full_slices,
    .repaired_partial_points = repair_stats.repaired_partial_points,
    .total_slices = extraction.total_slices,          // Nσ × Nr
    .total_points = extraction.total_slices * Nt      // Nσ × Nr × Nt
};
```

### Data Flow Summary

```
solve_batch() → batch_result.failed_count
                      ↓
extract_tensor() → ExtractionResult{tensor, total_slices, failed_pde, failed_spline}
                      ↓
repair_failed_slices() → RepairStats{repaired_full_slices, repaired_partial_slices}
                      ↓
fit_coeffs() → FitCoeffsResult (NaN-free, fitting succeeds)
                      ↓
build() → PriceTableResult{..., failed_*, repaired_*, ...}
```

### Implementation Order

1. Change `extract_tensor()` return type to `ExtractionResult<N>` with both failure vectors
2. Track both failure modes in parallel loop (mutex-protected push)
3. Add `repair_failed_slices()` with neighbor search and maturity interpolation
4. Change `build()` to accept partial failures (add `max_failure_rate` config)
5. Extend `PriceTableResult` with failure stats
6. Add tests for partial failure scenarios (PDE failures, spline failures, edge cases)

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
| 2b | Track both failure modes (PDE + spline) in parallel loop | Low | 2a, 1b |
| 2c | Add repair_failed_slices() with neighbor search | Medium | 2a |
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
