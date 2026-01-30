<!-- SPDX-License-Identifier: MIT -->
# PriceTableBuilder Phase 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add OpenMP parallelization to extract_tensor() and partial failure tolerance with repair to PriceTableBuilder.

**Architecture:** Parallelize the (σ,r) loop in extract_tensor() using MANGO_PRAGMA macros. Track PDE and spline failures separately, repair via neighbor interpolation, and surface statistics in PriceTableResult.

**Tech Stack:** C++23, Bazel, OpenMP via MANGO_PRAGMA_* macros, GoogleTest

**Design Reference:** `docs/plans/2025-11-24-price-table-builder-phase2.md`

---

## Task 1: Add OpenMP flags to BUILD.bazel

**Files:**
- Modify: `src/option/BUILD.bazel:55-74`

**Step 1: Add OpenMP copts and linkopts**

```python
# In src/option/BUILD.bazel, find price_table_builder target and add:
cc_library(
    name = "price_table_builder",
    srcs = ["price_table_builder.cpp"],
    hdrs = ["price_table_builder.hpp"],
    copts = [
        "-fopenmp",
        "-pthread",
    ],
    linkopts = [
        "-fopenmp",
        "-pthread",
    ],
    deps = [
        ":option_chain",
        ":price_table_config",
        ":price_table_axes",
        ":price_table_surface",
        ":price_tensor",
        ":american_option",
        ":american_option_batch",
        ":recursion_helpers",
        "//src/math:cubic_spline_solver",
        "//src/math:bspline_nd_separable",
        "//src/support:aligned_arena",
        "//src/support:ivcalc_trace_hdr",
        "//src/support:parallel",  # NEW: for MANGO_PRAGMA_* macros
    ],
    visibility = ["//visibility:public"],
)
```

**Step 2: Verify build succeeds**

Run: `bazel build //src/option:price_table_builder`
Expected: BUILD SUCCESS

**Step 3: Commit**

```bash
git add src/option/BUILD.bazel
git commit -m "build: add OpenMP flags to price_table_builder target"
```

---

## Task 2: Add includes for parallelization

**Files:**
- Modify: `src/option/price_table_builder.cpp:1-20`

**Step 1: Add required includes**

```cpp
// At top of src/option/price_table_builder.cpp, add:
#include "src/support/parallel.hpp"
#include <mutex>
#include <tuple>
#include <map>
#include <unordered_set>
```

**Step 2: Verify build succeeds**

Run: `bazel build //src/option:price_table_builder`
Expected: BUILD SUCCESS

**Step 3: Commit**

```bash
git add src/option/price_table_builder.cpp
git commit -m "refactor: add includes for OpenMP parallelization"
```

---

## Task 3: Add max_failure_rate to PriceTableConfig

**Files:**
- Modify: `src/option/price_table_config.hpp`
- Test: `tests/price_table_builder_test.cc`

**Step 1: Write failing test**

```cpp
// In tests/price_table_builder_test.cc, add:
TEST(PriceTableConfigTest, MaxFailureRateDefault) {
    mango::PriceTableConfig config;
    EXPECT_DOUBLE_EQ(config.max_failure_rate, 0.0);
}

TEST(PriceTableConfigTest, ValidateConfigRejectsInvalidRate) {
    mango::PriceTableConfig config;
    config.max_failure_rate = 1.5;  // Invalid
    auto err = mango::validate_config(config);
    EXPECT_TRUE(err.has_value());
    EXPECT_NE(err->find("max_failure_rate"), std::string::npos);
}

TEST(PriceTableConfigTest, ValidateConfigAcceptsValidRate) {
    mango::PriceTableConfig config;
    config.max_failure_rate = 0.1;  // Valid
    auto err = mango::validate_config(config);
    EXPECT_FALSE(err.has_value());
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:price_table_builder_test --test_filter="*MaxFailureRate*" --test_output=all`
Expected: FAIL - max_failure_rate not defined, validate_config not defined

**Step 3: Add max_failure_rate and validate_config**

```cpp
// In src/option/price_table_config.hpp, add include:
#include <string>
#include <optional>

// In struct PriceTableConfig, add field:
    double max_failure_rate = 0.0;  // 0.0 = strict, 0.1 = allow 10%

// After struct, add validation helper:
inline std::optional<std::string> validate_config(const PriceTableConfig& config) {
    if (config.max_failure_rate < 0.0 || config.max_failure_rate > 1.0) {
        return "max_failure_rate must be in [0.0, 1.0], got " +
               std::to_string(config.max_failure_rate);
    }
    return std::nullopt;
}
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:price_table_builder_test --test_filter="*MaxFailureRate*" --test_output=all`
Expected: PASS

**Step 5: Run all tests**

Run: `bazel test //tests:price_table_builder_test --test_output=errors`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/option/price_table_config.hpp tests/price_table_builder_test.cc
git commit -m "feat: add max_failure_rate config with validation"
```

---

## Task 4: Add validate_config call in build()

**Files:**
- Modify: `src/option/price_table_builder.cpp` (build method)
- Test: `tests/price_table_builder_test.cc`

**Step 1: Write failing test**

```cpp
// In tests/price_table_builder_test.cc, add:
TEST(PriceTableBuilderTest, BuildRejectsInvalidConfig) {
    mango::PriceTableConfig config;
    config.max_failure_rate = 2.0;  // Invalid
    config.option_type = mango::OptionType::PUT;
    config.K_ref = 100.0;

    mango::PriceTableBuilder<4> builder(config);

    // Create minimal valid axes
    auto axes_result = mango::PriceTableAxes<4>::from_vectors(
        {0.9, 1.0, 1.1}, {0.25}, {0.2}, {0.05});
    ASSERT_TRUE(axes_result.has_value());

    auto result = builder.build(axes_result.value());
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("max_failure_rate"), std::string::npos);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:price_table_builder_test --test_filter="*RejectsInvalidConfig*" --test_output=all`
Expected: FAIL - build() doesn't validate config

**Step 3: Add validation at start of build()**

```cpp
// In src/option/price_table_builder.cpp, at start of build() method:
template <size_t N>
std::expected<PriceTableResult<N>, std::string>
PriceTableBuilder<N>::build(const PriceTableAxes<N>& axes) {
    // Validate config
    if (auto err = validate_config(config_); err.has_value()) {
        return std::unexpected("Invalid config: " + err.value());
    }
    // ... rest of method
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:price_table_builder_test --test_filter="*RejectsInvalidConfig*" --test_output=all`
Expected: PASS

**Step 5: Run all tests**

Run: `bazel test //tests:price_table_builder_test --test_output=errors`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/option/price_table_builder.cpp tests/price_table_builder_test.cc
git commit -m "feat: validate config at start of build()"
```

---

## Task 5: Add validate_config to factory methods

**Files:**
- Modify: `src/option/price_table_builder.cpp` (from_vectors, from_strikes, from_chain)
- Test: `tests/price_table_builder_test.cc`

**Step 1: Write failing test**

```cpp
// In tests/price_table_builder_test.cc, add:
TEST(PriceTableBuilderTest, FromVectorsRejectsInvalidMaxFailureRate) {
    auto result = mango::PriceTableBuilder<4>::from_vectors(
        {0.9, 1.0, 1.1},  // moneyness
        {0.25, 0.5},      // maturity
        {0.2, 0.3},       // volatility
        {0.05},           // rate
        100.0,            // K_ref
        mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value(),
        500,              // n_time
        mango::OptionType::PUT,
        0.0,              // dividend_yield
        1.5               // max_failure_rate - INVALID
    );
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("max_failure_rate"), std::string::npos);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:price_table_builder_test --test_filter="*FromVectorsRejectsInvalid*" --test_output=all`
Expected: FAIL - from_vectors doesn't accept max_failure_rate parameter

**Step 3: Add max_failure_rate parameter to factories**

```cpp
// In src/option/price_table_builder.hpp, update from_vectors signature:
static std::expected<std::pair<PriceTableBuilder<4>, PriceTableAxes<4>>, std::string>
from_vectors(
    std::vector<double> moneyness,
    std::vector<double> maturity,
    std::vector<double> volatility,
    std::vector<double> rate,
    double K_ref,
    GridSpec<double> grid_spec,
    size_t n_time,
    OptionType type = OptionType::PUT,
    double dividend_yield = 0.0,
    double max_failure_rate = 0.0);  // NEW

// In src/option/price_table_builder.cpp, update implementation:
// After populating config, add:
config.max_failure_rate = max_failure_rate;
if (auto err = validate_config(config); err.has_value()) {
    return std::unexpected(err.value());
}

// Same pattern for from_strikes and from_chain
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:price_table_builder_test --test_filter="*FromVectorsRejectsInvalid*" --test_output=all`
Expected: PASS

**Step 5: Run all tests**

Run: `bazel test //tests:price_table_builder_test --test_output=errors`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/option/price_table_builder.hpp src/option/price_table_builder.cpp tests/price_table_builder_test.cc
git commit -m "feat: add max_failure_rate param to factory methods"
```

---

## Task 6: Define ExtractionResult struct

**Files:**
- Modify: `src/option/price_table_builder.hpp`

**Step 1: Add ExtractionResult definition**

```cpp
// In src/option/price_table_builder.hpp, before PriceTableBuilder class:
template <size_t N>
struct ExtractionResult {
    PriceTensor<N> tensor;
    size_t total_slices;
    std::vector<size_t> failed_pde;
    std::vector<std::tuple<size_t, size_t, size_t>> failed_spline;
};
```

**Step 2: Verify build succeeds**

Run: `bazel build //src/option:price_table_builder`
Expected: BUILD SUCCESS

**Step 3: Commit**

```bash
git add src/option/price_table_builder.hpp
git commit -m "feat: add ExtractionResult struct for failure tracking"
```

---

## Task 7: Change extract_tensor return type

**Files:**
- Modify: `src/option/price_table_builder.hpp` (declaration)
- Modify: `src/option/price_table_builder.cpp` (implementation)

**Step 1: Update declaration**

```cpp
// In src/option/price_table_builder.hpp, change:
[[nodiscard]] std::expected<ExtractionResult<N>, std::string> extract_tensor(
    const BatchAmericanOptionResult& batch,
    const PriceTableAxes<N>& axes) const;

// Also update extract_tensor_for_testing
[[nodiscard]] std::expected<ExtractionResult<N>, std::string> extract_tensor_for_testing(
    const BatchAmericanOptionResult& batch,
    const PriceTableAxes<N>& axes) const {
    return extract_tensor(batch, axes);
}
```

**Step 2: Update implementation to return ExtractionResult**

```cpp
// In src/option/price_table_builder.cpp, change extract_tensor:
template <size_t N>
std::expected<ExtractionResult<N>, std::string>
PriceTableBuilder<N>::extract_tensor(
    const BatchAmericanOptionResult& batch,
    const PriceTableAxes<N>& axes) const
{
    // ... existing tensor creation code ...

    // At end, return ExtractionResult instead of just tensor:
    return ExtractionResult<N>{
        .tensor = std::move(tensor),
        .total_slices = Nσ * Nr,
        .failed_pde = {},      // Will populate in Task 8
        .failed_spline = {}    // Will populate in Task 8
    };
}
```

**Step 3: Update build() to use new return type**

```cpp
// In build(), change:
auto extraction = extract_tensor(batch_result, axes);
if (!extraction.has_value()) {
    return std::unexpected(extraction.error());
}
// Use extraction->tensor for fit_coeffs
auto coeffs_result = fit_coeffs(extraction->tensor, axes);
```

**Step 4: Verify build succeeds**

Run: `bazel build //src/option:price_table_builder`
Expected: BUILD SUCCESS

**Step 5: Run all tests**

Run: `bazel test //tests:price_table_builder_test --test_output=errors`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/option/price_table_builder.hpp src/option/price_table_builder.cpp
git commit -m "refactor: change extract_tensor to return ExtractionResult"
```

---

## Task 8: Track failures in extract_tensor

**Files:**
- Modify: `src/option/price_table_builder.cpp` (extract_tensor)
- Test: `tests/price_table_builder_test.cc`

**Step 1: Write failing test**

```cpp
// In tests/price_table_builder_test.cc, add:
TEST(PriceTableBuilderTest, ExtractTensorTracksPDEFailures) {
    // This test requires mocking batch results with failures
    // For now, verify the fields exist and are initialized
    mango::PriceTableConfig config;
    config.K_ref = 100.0;
    mango::PriceTableBuilder<4> builder(config);

    auto axes_result = mango::PriceTableAxes<4>::from_vectors(
        {0.9, 1.0, 1.1}, {0.25}, {0.2}, {0.05});
    ASSERT_TRUE(axes_result.has_value());

    // Create batch with all successes
    std::vector<mango::AmericanOptionParams> batch_params;
    // ... populate minimal batch ...

    // For now just verify ExtractionResult has the expected fields
    // Full failure tracking test after implementation
}
```

**Step 2: Add mutex and failure tracking to extract_tensor**

```cpp
// In extract_tensor(), add after tensor creation:
std::vector<size_t> failed_pde;
std::vector<std::tuple<size_t, size_t, size_t>> failed_spline;
std::mutex failed_mutex;

// In the main loop, add failure tracking:
MANGO_PRAGMA_PARALLEL
{
    MANGO_PRAGMA_FOR_COLLAPSE2
    for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
        for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
            size_t batch_idx = σ_idx * Nr + r_idx;

            if (!batch.results[batch_idx].has_value()) {
                {
                    std::lock_guard<std::mutex> lock(failed_mutex);
                    failed_pde.push_back(batch_idx);
                }
                // Fill with NaN (existing code)
                continue;
            }

            // Per-maturity spline tracking
            for (size_t τ_idx = 0; τ_idx < Nt; ++τ_idx) {
                // ... existing spline code ...
                if (err.has_value()) {
                    {
                        std::lock_guard<std::mutex> lock(failed_mutex);
                        failed_spline.emplace_back(σ_idx, r_idx, τ_idx);
                    }
                    // Fill with NaN
                    continue;
                }
            }
        }
    }
}

// Return with populated failure vectors
return ExtractionResult<N>{
    .tensor = std::move(tensor),
    .total_slices = Nσ * Nr,
    .failed_pde = std::move(failed_pde),
    .failed_spline = std::move(failed_spline)
};
```

**Step 3: Verify build succeeds**

Run: `bazel build //src/option:price_table_builder`
Expected: BUILD SUCCESS

**Step 4: Run all tests**

Run: `bazel test //tests:price_table_builder_test --test_output=errors`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/option/price_table_builder.cpp tests/price_table_builder_test.cc
git commit -m "feat: track PDE and spline failures in extract_tensor"
```

---

## Task 9: Add RepairStats and repair_failed_slices declaration

**Files:**
- Modify: `src/option/price_table_builder.hpp`

**Step 1: Add RepairStats struct and function declaration**

```cpp
// In src/option/price_table_builder.hpp, add:
struct RepairStats {
    size_t repaired_full_slices;
    size_t repaired_partial_points;
};

// In PriceTableBuilder private section, add:
[[nodiscard]] std::expected<RepairStats, std::string> repair_failed_slices(
    PriceTensor<N>& tensor,
    const std::vector<size_t>& failed_pde,
    const std::vector<std::tuple<size_t, size_t, size_t>>& failed_spline,
    const PriceTableAxes<N>& axes) const;

[[nodiscard]] std::optional<std::pair<size_t, size_t>> find_nearest_valid_neighbor(
    size_t σ_idx, size_t r_idx, size_t Nσ, size_t Nr,
    const std::vector<bool>& slice_valid) const;
```

**Step 2: Verify build succeeds**

Run: `bazel build //src/option:price_table_builder`
Expected: BUILD SUCCESS (linker error expected until implementation)

**Step 3: Commit**

```bash
git add src/option/price_table_builder.hpp
git commit -m "feat: add RepairStats and repair function declarations"
```

---

## Task 10: Implement find_nearest_valid_neighbor

**Files:**
- Modify: `src/option/price_table_builder.cpp`
- Test: `tests/price_table_builder_test.cc`

**Step 1: Write failing test**

```cpp
// In tests/price_table_builder_test.cc, add:
TEST(PriceTableBuilderTest, FindNearestValidNeighborFindsAdjacent) {
    // Test helper directly via builder's testing interface
    // Create 3x3 grid, mark center invalid, verify finds adjacent
    std::vector<bool> slice_valid(9, true);
    slice_valid[4] = false;  // Center (1,1) invalid

    mango::PriceTableConfig config;
    mango::PriceTableBuilder<4> builder(config);

    auto result = builder.find_nearest_valid_neighbor_for_testing(1, 1, 3, 3, slice_valid);
    ASSERT_TRUE(result.has_value());
    // Should find one of (0,1), (1,0), (1,2), (2,1) at distance 1
    auto [nσ, nr] = result.value();
    size_t dist = std::abs(static_cast<int>(nσ) - 1) + std::abs(static_cast<int>(nr) - 1);
    EXPECT_EQ(dist, 1);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:price_table_builder_test --test_filter="*FindNearestValid*" --test_output=all`
Expected: FAIL - function not implemented

**Step 3: Implement find_nearest_valid_neighbor**

```cpp
// In src/option/price_table_builder.cpp, add:
template <size_t N>
std::optional<std::pair<size_t, size_t>>
PriceTableBuilder<N>::find_nearest_valid_neighbor(
    size_t σ_idx, size_t r_idx, size_t Nσ, size_t Nr,
    const std::vector<bool>& slice_valid) const
{
    const size_t max_dist = (Nσ - 1) + (Nr - 1);

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
    return std::nullopt;
}
```

**Step 4: Add testing interface to header**

```cpp
// In src/option/price_table_builder.hpp, public section:
[[nodiscard]] std::optional<std::pair<size_t, size_t>> find_nearest_valid_neighbor_for_testing(
    size_t σ_idx, size_t r_idx, size_t Nσ, size_t Nr,
    const std::vector<bool>& slice_valid) const {
    return find_nearest_valid_neighbor(σ_idx, r_idx, Nσ, Nr, slice_valid);
}
```

**Step 5: Run test to verify it passes**

Run: `bazel test //tests:price_table_builder_test --test_filter="*FindNearestValid*" --test_output=all`
Expected: PASS

**Step 6: Commit**

```bash
git add src/option/price_table_builder.hpp src/option/price_table_builder.cpp tests/price_table_builder_test.cc
git commit -m "feat: implement find_nearest_valid_neighbor"
```

---

## Task 11: Implement repair_failed_slices

**Files:**
- Modify: `src/option/price_table_builder.cpp`
- Test: `tests/price_table_builder_test.cc`

**Step 1: Write failing test**

```cpp
// In tests/price_table_builder_test.cc, add:
TEST(PriceTableBuilderTest, RepairFailedSlicesInterpolatesPartial) {
    // Create tensor with one NaN at τ=1, verify τ-interpolation fills it
    // Details depend on PriceTensor API - implement after Task 10
}

TEST(PriceTableBuilderTest, RepairFailedSlicesCopiesFromNeighbor) {
    // Create tensor with full slice NaN, verify neighbor copy
}

TEST(PriceTableBuilderTest, RepairFailedSlicesFailsWhenNoValidDonor) {
    // All slices invalid, verify returns error
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:price_table_builder_test --test_filter="*RepairFailedSlices*" --test_output=all`
Expected: FAIL - function not implemented

**Step 3: Implement repair_failed_slices**

See design document `docs/plans/2025-11-24-price-table-builder-phase2.md` lines 322-479 for complete implementation.

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:price_table_builder_test --test_filter="*RepairFailedSlices*" --test_output=all`
Expected: PASS

**Step 5: Commit**

```bash
git add src/option/price_table_builder.cpp tests/price_table_builder_test.cc
git commit -m "feat: implement repair_failed_slices with two-phase repair"
```

---

## Task 12: Extend PriceTableResult with failure stats

**Files:**
- Modify: `src/option/price_table_builder.hpp`

**Step 1: Add fields to PriceTableResult**

```cpp
// In src/option/price_table_builder.hpp, PriceTableResult struct:
template <size_t N>
struct PriceTableResult {
    std::shared_ptr<const PriceTableSurface<N>> surface = nullptr;
    size_t n_pde_solves = 0;
    double precompute_time_seconds = 0.0;
    BSplineFittingStats fitting_stats;
    // NEW: Failure and repair tracking
    size_t failed_pde_slices = 0;
    size_t failed_spline_points = 0;
    size_t repaired_full_slices = 0;
    size_t repaired_partial_points = 0;
    size_t total_slices = 0;
    size_t total_points = 0;
};
```

**Step 2: Verify build succeeds**

Run: `bazel build //src/option:price_table_builder`
Expected: BUILD SUCCESS

**Step 3: Commit**

```bash
git add src/option/price_table_builder.hpp
git commit -m "feat: add failure stats to PriceTableResult"
```

---

## Task 13: Wire repair into build() and populate stats

**Files:**
- Modify: `src/option/price_table_builder.cpp` (build method)
- Test: `tests/price_table_builder_test.cc`

**Step 1: Write test for stats population**

```cpp
// In tests/price_table_builder_test.cc, add:
TEST(PriceTableBuilderTest, BuildPopulatesTotalSlicesAndPoints) {
    auto result = mango::PriceTableBuilder<4>::from_vectors(
        {0.9, 1.0, 1.1}, {0.25, 0.5}, {0.2}, {0.05},
        100.0,
        mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value(),
        500);
    ASSERT_TRUE(result.has_value());
    auto& [builder, axes] = result.value();

    auto build_result = builder.build(axes);
    ASSERT_TRUE(build_result.has_value());

    EXPECT_EQ(build_result->total_slices, 1 * 1);  // Nσ × Nr = 1 × 1
    EXPECT_EQ(build_result->total_points, 1 * 1 * 2);  // Nσ × Nr × Nt
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:price_table_builder_test --test_filter="*PopulatesTotalSlices*" --test_output=all`
Expected: FAIL - fields not populated

**Step 3: Update build() to call repair and populate stats**

```cpp
// In build(), after extract_tensor:
auto repair_result = repair_failed_slices(
    extraction->tensor, extraction->failed_pde, extraction->failed_spline, axes);
if (!repair_result.has_value()) {
    return std::unexpected(repair_result.error());
}
auto repair_stats = repair_result.value();

// ... fit_coeffs ...

// At return:
const size_t Nt = axes.grids[1].size();
return PriceTableResult<N>{
    .surface = std::move(surface),
    .n_pde_solves = batch_result.results.size(),
    .precompute_time_seconds = elapsed,
    .fitting_stats = coeffs_result->stats,
    .failed_pde_slices = extraction->failed_pde.size(),
    .failed_spline_points = extraction->failed_spline.size(),
    .repaired_full_slices = repair_stats.repaired_full_slices,
    .repaired_partial_points = repair_stats.repaired_partial_points,
    .total_slices = extraction->total_slices,
    .total_points = extraction->total_slices * Nt
};
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:price_table_builder_test --test_filter="*PopulatesTotalSlices*" --test_output=all`
Expected: PASS

**Step 5: Run all tests**

Run: `bazel test //... --test_output=errors`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/option/price_table_builder.cpp tests/price_table_builder_test.cc
git commit -m "feat: wire repair into build() and populate failure stats"
```

---

## Task 14: Final verification

**Step 1: Run all tests**

Run: `bazel test //...`
Expected: All tests PASS

**Step 2: Build all targets**

Run: `bazel build //...`
Expected: BUILD SUCCESS

**Step 3: Run benchmarks to verify no regression**

Run: `bazel run //benchmarks:readme_benchmarks -- --benchmark_filter="PriceTable"`
Expected: Performance comparable or better

**Step 4: Create summary commit**

```bash
git log --oneline -15  # Review all commits
```

---

## Verification Checklist

- [ ] All tests pass: `bazel test //...`
- [ ] All examples build: `bazel build //examples/...`
- [ ] All benchmarks build: `bazel build //benchmarks/...`
- [ ] No compiler warnings
- [ ] max_failure_rate validated in factories and build()
- [ ] extract_tensor parallelized with MANGO_PRAGMA macros
- [ ] PDE and spline failures tracked separately
- [ ] repair_failed_slices handles both failure types
- [ ] PriceTableResult includes failure statistics
