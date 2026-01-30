<!-- SPDX-License-Identifier: MIT -->
# ThreadWorkspaceBuffer Implementation Plan

**For Claude: REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute this plan**

## Overview

This plan implements the ThreadWorkspaceBuffer design from `docs/plans/2025-11-27-thread-workspace-buffer-design.md`. The goal is to eliminate ~24,000 memory allocations in B-spline fitting hot paths by introducing reusable per-thread byte buffers.

**Target:** Linux/glibc + C++23 (GCC 14+ / Clang 18+)

---

## Phase 0: Parallel Primitives

### Task 0.1: Add MANGO_PRAGMA_CRITICAL macro

**File:** `src/support/parallel.hpp`

**Test (RED):**
```cpp
// tests/parallel_critical_test.cc
#include "src/support/parallel.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <vector>

TEST(ParallelCriticalTest, AtomicCounterCorrectness) {
    std::atomic<int> counter{0};
    const int iterations = 1000;

    MANGO_PRAGMA_PARALLEL
    {
        MANGO_PRAGMA_FOR_STATIC
        for (int i = 0; i < iterations; ++i) {
            MANGO_PRAGMA_CRITICAL
            {
                ++counter;
            }
        }
    }

    EXPECT_EQ(counter.load(), iterations);
}
```

**Implementation:**
Add to `src/support/parallel.hpp` after line 43:
```cpp
#if defined(_OPENMP)
    #define MANGO_PRAGMA_CRITICAL _Pragma("omp critical")
#else
    #define MANGO_PRAGMA_CRITICAL
#endif
```

**Verification:** `bazel test //tests:parallel_critical_test`

---

### Task 0.2: Add start_array_lifetime helper

**File:** `src/support/lifetime.hpp` (NEW)

**Test (RED):**
```cpp
// tests/lifetime_test.cc
#include "src/support/lifetime.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <type_traits>

using namespace mango;

TEST(LifetimeTest, StartArrayLifetimeDouble) {
    alignas(64) std::byte buffer[sizeof(double) * 10];

    double* arr = start_array_lifetime<double>(buffer, 10);

    EXPECT_NE(arr, nullptr);
    // Write and read back
    for (size_t i = 0; i < 10; ++i) {
        arr[i] = static_cast<double>(i);
    }
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(arr[i], static_cast<double>(i));
    }
}

TEST(LifetimeTest, StartArrayLifetimeInt) {
    alignas(64) std::byte buffer[sizeof(int) * 10];

    int* arr = start_array_lifetime<int>(buffer, 10);

    EXPECT_NE(arr, nullptr);
    for (size_t i = 0; i < 10; ++i) {
        arr[i] = static_cast<int>(i);
    }
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(arr[i], static_cast<int>(i));
    }
}

// Verify static_assert fires for non-trivially-destructible types
// (This is a compile-time check - if this compiles, the assert works)
struct NonTrivial {
    ~NonTrivial() {}  // Non-trivial destructor
};
static_assert(!std::is_trivially_destructible_v<NonTrivial>);
// Uncommenting below should fail to compile:
// auto* bad = start_array_lifetime<NonTrivial>(nullptr, 0);
```

**Implementation:**
```cpp
// src/support/lifetime.hpp
#pragma once

#include <cstddef>
#include <type_traits>
#include <memory>  // std::start_lifetime_as_array (C++23)

namespace mango {

/// Start lifetime of T[n] array at given memory location
///
/// Uses std::start_lifetime_as_array (C++23) to properly create objects.
/// This is NOT equivalent to std::launder - it actually starts object lifetimes.
///
/// IMPORTANT: T must be trivially destructible because no destructor is ever
/// called when the workspace goes out of scope.
///
/// @tparam T Element type (must be trivially destructible)
/// @param p Pointer to suitably aligned memory
/// @param n Number of elements
/// @return Pointer to first element of the array
template<typename T>
T* start_array_lifetime(void* p, size_t n) {
    static_assert(std::is_trivially_destructible_v<T>,
        "start_array_lifetime requires trivially destructible types because "
        "no destructor is called when the workspace goes out of scope");
    return std::start_lifetime_as_array<T>(p, n);
}

/// Align offset up to specified alignment
constexpr size_t align_up(size_t offset, size_t alignment) noexcept {
    return (offset + alignment - 1) & ~(alignment - 1);
}

}  // namespace mango
```

**Verification:** `bazel test //tests:lifetime_test`

---

### Task 0.3: Create ThreadWorkspaceBuffer class

**File:** `src/support/thread_workspace.hpp` (NEW)

**Test (RED):**
```cpp
// tests/thread_workspace_test.cc
#include "src/support/thread_workspace.hpp"
#include <gtest/gtest.h>
#include <cstdint>

using namespace mango;

TEST(ThreadWorkspaceBufferTest, BasicAllocation) {
    ThreadWorkspaceBuffer buffer(1024);

    EXPECT_GE(buffer.size(), 1024u);
    EXPECT_EQ(buffer.size() % 64, 0u);  // 64-byte aligned size
}

TEST(ThreadWorkspaceBufferTest, Alignment64Byte) {
    ThreadWorkspaceBuffer buffer(100);

    auto bytes = buffer.bytes();
    auto addr = reinterpret_cast<std::uintptr_t>(bytes.data());
    EXPECT_EQ(addr % 64, 0u) << "Base address must be 64-byte aligned";
}

TEST(ThreadWorkspaceBufferTest, ByteSpanStability) {
    ThreadWorkspaceBuffer buffer(512);

    auto span1 = buffer.bytes();
    auto span2 = buffer.bytes();

    EXPECT_EQ(span1.data(), span2.data());
    EXPECT_EQ(span1.size(), span2.size());
}

TEST(ThreadWorkspaceBufferTest, PMRResourceAccessible) {
    ThreadWorkspaceBuffer buffer(1024);

    std::pmr::memory_resource& resource = buffer.resource();

    // Should be able to allocate from the resource
    void* p = resource.allocate(64, 8);
    EXPECT_NE(p, nullptr);
    resource.deallocate(p, 64, 8);
}

TEST(ThreadWorkspaceBufferTest, MoveSemantics) {
    ThreadWorkspaceBuffer buffer1(512);
    auto* original_data = buffer1.bytes().data();

    ThreadWorkspaceBuffer buffer2(std::move(buffer1));

    EXPECT_EQ(buffer2.bytes().data(), original_data);
}
```

**Implementation:**
Create `src/support/thread_workspace.hpp` with the exact implementation from the design document (lines 58-173).

**Verification:** `bazel test //tests:thread_workspace_test`

---

### Task 0.4: Add ThreadWorkspaceBuffer to BUILD file

**File:** `src/support/BUILD.bazel`

Add library target:
```python
cc_library(
    name = "thread_workspace",
    hdrs = ["thread_workspace.hpp", "lifetime.hpp"],
    deps = [":parallel"],
    visibility = ["//visibility:public"],
)
```

Add test target to `tests/BUILD.bazel`:
```python
cc_test(
    name = "thread_workspace_test",
    srcs = ["thread_workspace_test.cc"],
    deps = [
        "//src/support:thread_workspace",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "lifetime_test",
    srcs = ["lifetime_test.cc"],
    deps = [
        "//src/support:thread_workspace",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "parallel_critical_test",
    srcs = ["parallel_critical_test.cc"],
    deps = [
        "//src/support:parallel",
        "@googletest//:gtest_main",
    ],
)
```

**Verification:** `bazel build //src/support:thread_workspace && bazel test //tests:thread_workspace_test //tests:lifetime_test //tests:parallel_critical_test`

---

## Phase 1: B-Spline Collocation Workspace

### Task 1.1: Extend InterpolationError for workspace failures

**File:** `src/support/error_types.hpp`

**Test (RED):**
```cpp
// tests/interpolation_error_test.cc (add to existing or create)
#include "src/support/error_types.hpp"
#include <gtest/gtest.h>
#include <sstream>

using namespace mango;

TEST(InterpolationErrorTest, WorkspaceCreationFailedCode) {
    InterpolationError err(InterpolationErrorCode::WorkspaceCreationFailed,
                           "Buffer too small: 100 < 200 required");

    EXPECT_EQ(err.code, InterpolationErrorCode::WorkspaceCreationFailed);
    EXPECT_EQ(err.message, "Buffer too small: 100 < 200 required");
}

TEST(InterpolationErrorTest, BackwardCompatibleConstructor) {
    // Old-style construction should still work
    InterpolationError err(InterpolationErrorCode::FittingFailed, 100, 5, 0.001);

    EXPECT_EQ(err.code, InterpolationErrorCode::FittingFailed);
    EXPECT_EQ(err.grid_size, 100u);
    EXPECT_EQ(err.index, 5u);
    EXPECT_DOUBLE_EQ(err.max_residual, 0.001);
    EXPECT_TRUE(err.message.empty());
}

TEST(InterpolationErrorTest, OutputStreamWithMessage) {
    InterpolationError err(InterpolationErrorCode::WorkspaceCreationFailed,
                           "Test message");

    std::ostringstream oss;
    oss << err;
    std::string output = oss.str();

    EXPECT_NE(output.find("WorkspaceCreationFailed"), std::string::npos);
    EXPECT_NE(output.find("Test message"), std::string::npos);
}

TEST(InterpolationErrorTest, ConvertToPriceTableError) {
    InterpolationError err(InterpolationErrorCode::WorkspaceCreationFailed,
                           "Buffer allocation failed");

    PriceTableError pte = convert_to_price_table_error(err);

    EXPECT_EQ(pte.code, PriceTableErrorCode::ArenaAllocationFailed);
}
```

**Implementation:**

1. Add enum value to `InterpolationErrorCode` (after line 90):
```cpp
    ExtrapolationNotAllowed,
    WorkspaceCreationFailed   // NEW: from_bytes() failed
```

2. Add message field to `InterpolationError` struct (line 94-105):
```cpp
struct InterpolationError {
    InterpolationErrorCode code;
    size_t grid_size;
    size_t index;
    double max_residual;
    std::string message;  // NEW: empty for most errors

    // Existing constructor (backward compatible)
    InterpolationError(InterpolationErrorCode code,
                      size_t grid_size = 0,
                      size_t index = 0,
                      double max_residual = 0.0)
        : code(code), grid_size(grid_size), index(index), max_residual(max_residual) {}

    // NEW: Constructor with message for workspace errors
    InterpolationError(InterpolationErrorCode code, std::string msg)
        : code(code), grid_size(0), index(0), max_residual(0.0), message(std::move(msg)) {}
};
```

3. Update `operator<<` (line 223-229):
```cpp
inline std::ostream& operator<<(std::ostream& os, const InterpolationError& err) {
    os << "InterpolationError{code=" << static_cast<int>(err.code)
       << ", grid_size=" << err.grid_size
       << ", index=" << err.index
       << ", max_residual=" << err.max_residual;
    if (!err.message.empty()) {
        os << ", message=\"" << err.message << "\"";
    }
    os << "}";
    return os;
}
```

4. Update `convert_to_price_table_error` switch (line 309-332):
```cpp
inline PriceTableError convert_to_price_table_error(const InterpolationError& err) {
    PriceTableErrorCode code;
    switch (err.code) {
        case InterpolationErrorCode::InsufficientGridPoints:
            code = PriceTableErrorCode::InsufficientGridPoints;
            break;
        case InterpolationErrorCode::GridNotSorted:
        case InterpolationErrorCode::ZeroWidthGrid:
            code = PriceTableErrorCode::GridNotSorted;
            break;
        case InterpolationErrorCode::ValueSizeMismatch:
        case InterpolationErrorCode::BufferSizeMismatch:
        case InterpolationErrorCode::DimensionMismatch:
        case InterpolationErrorCode::CoefficientSizeMismatch:
        case InterpolationErrorCode::NaNInput:
        case InterpolationErrorCode::InfInput:
        case InterpolationErrorCode::FittingFailed:
        case InterpolationErrorCode::EvaluationFailed:
        case InterpolationErrorCode::ExtrapolationNotAllowed:
            code = PriceTableErrorCode::FittingFailed;
            break;
        case InterpolationErrorCode::WorkspaceCreationFailed:
            code = PriceTableErrorCode::ArenaAllocationFailed;
            break;
    }
    return PriceTableError{code, err.index, err.grid_size};
}
```

**Verification:** `bazel test //tests:interpolation_error_test` (or add to existing error_types tests)

---

### Task 1.2: Create BSplineCollocationWorkspace template

**File:** `src/math/bspline_collocation_workspace.hpp` (NEW)

**Test (RED):**
```cpp
// tests/bspline_collocation_workspace_test.cc
#include "src/math/bspline_collocation_workspace.hpp"
#include "src/support/thread_workspace.hpp"
#include <gtest/gtest.h>
#include <cstdint>

using namespace mango;

TEST(BSplineCollocationWorkspaceTest, RequiredBytesCalculation) {
    // For n=100, bandwidth=4:
    // band_storage: 10*100*8 = 8000 bytes + padding
    // lapack_storage: 10*100*8 = 8000 bytes + padding
    // pivots: 100*4 = 400 bytes + padding
    // coeffs: 100*8 = 800 bytes + padding
    size_t bytes = BSplineCollocationWorkspace<double>::required_bytes(100);

    // Should be at least sum of minimums
    EXPECT_GE(bytes, 8000u + 8000u + 400u + 800u);
    // Should be 64-byte aligned
    EXPECT_EQ(bytes % 64, 0u);
}

TEST(BSplineCollocationWorkspaceTest, FromBytesSuccess) {
    const size_t n = 50;
    ThreadWorkspaceBuffer buffer(BSplineCollocationWorkspace<double>::required_bytes(n));

    auto result = BSplineCollocationWorkspace<double>::from_bytes(buffer.bytes(), n);

    ASSERT_TRUE(result.has_value()) << "from_bytes failed";
    auto& ws = result.value();

    EXPECT_EQ(ws.size(), n);
    EXPECT_EQ(ws.band_storage().size(), 10 * n);  // LDAB=10
    EXPECT_EQ(ws.lapack_storage().size(), 10 * n);
    EXPECT_EQ(ws.pivots().size(), n);
    EXPECT_EQ(ws.coeffs().size(), n);
}

TEST(BSplineCollocationWorkspaceTest, BufferTooSmall) {
    const size_t n = 50;
    size_t required = BSplineCollocationWorkspace<double>::required_bytes(n);

    // Allocate less than required
    std::vector<std::byte> small_buffer(required / 2);

    auto result = BSplineCollocationWorkspace<double>::from_bytes(
        std::span(small_buffer), n);

    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("Buffer too small"), std::string::npos);
}

TEST(BSplineCollocationWorkspaceTest, SpansAre64ByteAligned) {
    const size_t n = 100;
    ThreadWorkspaceBuffer buffer(BSplineCollocationWorkspace<double>::required_bytes(n));

    auto ws = BSplineCollocationWorkspace<double>::from_bytes(buffer.bytes(), n).value();

    // All spans should start at 64-byte aligned addresses
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ws.band_storage().data()) % 64, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ws.lapack_storage().data()) % 64, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ws.pivots().data()) % 64, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ws.coeffs().data()) % 64, 0u);
}

TEST(BSplineCollocationWorkspaceTest, SpansNonOverlapping) {
    const size_t n = 50;
    ThreadWorkspaceBuffer buffer(BSplineCollocationWorkspace<double>::required_bytes(n));

    auto ws = BSplineCollocationWorkspace<double>::from_bytes(buffer.bytes(), n).value();

    auto* band_end = ws.band_storage().data() + ws.band_storage().size();
    auto* lapack_start = ws.lapack_storage().data();
    auto* lapack_end = ws.lapack_storage().data() + ws.lapack_storage().size();
    auto* pivots_start = ws.pivots().data();
    auto* pivots_end = reinterpret_cast<double*>(
        reinterpret_cast<std::byte*>(ws.pivots().data()) + ws.pivots().size() * sizeof(int));
    auto* coeffs_start = ws.coeffs().data();

    // band_storage < lapack_storage
    EXPECT_LE(reinterpret_cast<std::byte*>(band_end),
              reinterpret_cast<std::byte*>(lapack_start));
    // lapack_storage < pivots
    EXPECT_LE(reinterpret_cast<std::byte*>(lapack_end),
              reinterpret_cast<std::byte*>(pivots_start));
    // pivots < coeffs
    EXPECT_LE(reinterpret_cast<std::byte*>(pivots_end),
              reinterpret_cast<std::byte*>(coeffs_start));
}
```

**Implementation:**
Create `src/math/bspline_collocation_workspace.hpp` with the exact implementation from the design document (lines 193-333).

**Verification:** `bazel test //tests:bspline_collocation_workspace_test`

---

### Task 1.3: Add fit_with_workspace to BSplineCollocation1D

**File:** `src/math/bspline_collocation.hpp`

**Test (RED):**
```cpp
// tests/bspline_fit_with_workspace_test.cc
#include "src/math/bspline_collocation.hpp"
#include "src/math/bspline_collocation_workspace.hpp"
#include "src/support/thread_workspace.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace mango;

class BSplineFitWithWorkspaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create uniform grid
        const size_t n = 50;
        grid.resize(n);
        values.resize(n);
        for (size_t i = 0; i < n; ++i) {
            double t = static_cast<double>(i) / (n - 1);
            grid[i] = t;
            values[i] = std::sin(2 * M_PI * t);  // Sine wave
        }

        auto solver_result = BSplineCollocation1D<double>::create(grid);
        ASSERT_TRUE(solver_result.has_value());
        solver = std::make_unique<BSplineCollocation1D<double>>(
            std::move(solver_result.value()));
    }

    std::vector<double> grid;
    std::vector<double> values;
    std::unique_ptr<BSplineCollocation1D<double>> solver;
};

TEST_F(BSplineFitWithWorkspaceTest, ProducesSameResultAsFit) {
    // Fit with regular method
    auto result_regular = solver->fit(values);
    ASSERT_TRUE(result_regular.has_value());

    // Fit with workspace method
    ThreadWorkspaceBuffer buffer(
        BSplineCollocationWorkspace<double>::required_bytes(grid.size()));
    auto ws = BSplineCollocationWorkspace<double>::from_bytes(
        buffer.bytes(), grid.size()).value();

    auto result_ws = solver->fit_with_workspace(
        std::span<const double>(values), ws);
    ASSERT_TRUE(result_ws.has_value());

    // Results should match
    EXPECT_NEAR(result_regular->max_residual, result_ws->max_residual, 1e-14);
    EXPECT_NEAR(result_regular->condition_estimate, result_ws->condition_estimate, 1e-10);

    // Coefficients should match (workspace stores in ws.coeffs())
    ASSERT_EQ(result_regular->coefficients.size(), ws.coeffs().size());
    for (size_t i = 0; i < result_regular->coefficients.size(); ++i) {
        EXPECT_NEAR(result_regular->coefficients[i], ws.coeffs()[i], 1e-14)
            << "Coefficient mismatch at index " << i;
    }
}

TEST_F(BSplineFitWithWorkspaceTest, WorkspaceReusable) {
    ThreadWorkspaceBuffer buffer(
        BSplineCollocationWorkspace<double>::required_bytes(grid.size()));
    auto ws = BSplineCollocationWorkspace<double>::from_bytes(
        buffer.bytes(), grid.size()).value();

    // Fit multiple times with same workspace
    for (int iter = 0; iter < 3; ++iter) {
        // Modify values each iteration
        for (size_t i = 0; i < values.size(); ++i) {
            double t = static_cast<double>(i) / (values.size() - 1);
            values[i] = std::sin(2 * M_PI * t * (iter + 1));
        }

        auto result = solver->fit_with_workspace(
            std::span<const double>(values), ws);

        ASSERT_TRUE(result.has_value()) << "Fit failed on iteration " << iter;
        EXPECT_LT(result->max_residual, 1e-9);
    }
}

TEST_F(BSplineFitWithWorkspaceTest, ValueSizeMismatch) {
    ThreadWorkspaceBuffer buffer(
        BSplineCollocationWorkspace<double>::required_bytes(grid.size()));
    auto ws = BSplineCollocationWorkspace<double>::from_bytes(
        buffer.bytes(), grid.size()).value();

    // Wrong size values
    std::vector<double> wrong_size(grid.size() + 10, 1.0);

    auto result = solver->fit_with_workspace(
        std::span<const double>(wrong_size), ws);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, InterpolationErrorCode::ValueSizeMismatch);
}
```

**Implementation:**
Add to `src/math/bspline_collocation.hpp` after `fit_with_buffer()` method (after line 302):

```cpp
    /// Fit with external workspace (zero-allocation variant)
    ///
    /// Uses BSplineCollocationWorkspace for all temporary storage.
    /// Coefficients are written to ws.coeffs().
    ///
    /// @param values Function values at grid points (size n_)
    /// @param ws Pre-allocated workspace (must have size() == n_)
    /// @param config Solver configuration
    /// @return Fit result (coefficients are in ws.coeffs(), not in result)
    [[nodiscard]] std::expected<BSplineCollocationResult<T>, InterpolationError>
    fit_with_workspace(
        std::span<const T> values,
        BSplineCollocationWorkspace<T>& ws,
        const BSplineCollocationConfig<T>& config = {})
    {
        if (values.size() != n_) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::ValueSizeMismatch,
                values.size()});
        }
        if (ws.size() != n_) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::BufferSizeMismatch,
                ws.size()});
        }

        // Validate input values
        for (size_t i = 0; i < n_; ++i) {
            if (std::isnan(values[i])) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::NaNInput, n_, i});
            }
            if (std::isinf(values[i])) {
                return std::unexpected(InterpolationError{
                    InterpolationErrorCode::InfInput, n_, i});
            }
        }

        // Build collocation matrix into workspace band_storage
        build_collocation_matrix_to_workspace(ws);

        // Factorize using workspace lapack_storage and pivots
        auto factor_result = factorize_banded_workspace(ws);
        if (!factor_result.ok()) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed, n_});
        }

        // Solve into ws.coeffs()
        auto solve_result = solve_banded_workspace(ws, values);
        if (!solve_result.ok()) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed, n_});
        }

        // Compute residuals
        const T max_residual = compute_residual_from_span(ws.coeffs(), values);

        if (max_residual > config.tolerance) {
            return std::unexpected(InterpolationError{
                InterpolationErrorCode::FittingFailed, n_, 0,
                static_cast<double>(max_residual)});
        }

        // Estimate condition number
        const T norm_A = compute_matrix_norm1();
        const T cond_est = estimate_banded_condition_workspace(ws, norm_A);

        return BSplineCollocationResult<T>{
            .coefficients = {},  // Caller uses ws.coeffs()
            .max_residual = max_residual,
            .condition_estimate = cond_est
        };
    }
```

Also add private helper methods:
- `build_collocation_matrix_to_workspace(BSplineCollocationWorkspace<T>& ws)`
- `factorize_banded_workspace(BSplineCollocationWorkspace<T>& ws)`
- `solve_banded_workspace(BSplineCollocationWorkspace<T>& ws, std::span<const T> rhs)`
- `estimate_banded_condition_workspace(BSplineCollocationWorkspace<T>& ws, T norm_A)`

These reuse the existing logic but operate on workspace spans instead of allocating BandedMatrix/BandedLUWorkspace.

**Verification:** `bazel test //tests:bspline_fit_with_workspace_test`

---

### Task 1.4: Add BUILD targets for Phase 1

**File:** `src/math/BUILD.bazel`, `tests/BUILD.bazel`

Add library:
```python
cc_library(
    name = "bspline_collocation_workspace",
    hdrs = ["bspline_collocation_workspace.hpp"],
    deps = [
        "//src/support:lifetime",
        "//src/support:error_types",
    ],
    visibility = ["//visibility:public"],
)
```

Update `bspline_collocation` deps to include workspace.

Add tests to `tests/BUILD.bazel`.

**Verification:** `bazel build //src/math:bspline_collocation_workspace && bazel test //tests:bspline_collocation_workspace_test //tests:bspline_fit_with_workspace_test`

---

### Task 1.5: Integration test - BSplineNDSeparable with workspace

**File:** `tests/bspline_nd_workspace_integration_test.cc` (NEW)

**Test:**
```cpp
// Verify 4D tensor fitting produces identical results with and without workspace
#include "src/math/bspline_nd_separable.hpp"
#include "src/support/thread_workspace.hpp"
#include <gtest/gtest.h>

using namespace mango;

TEST(BSplineNDWorkspaceIntegrationTest, IdenticalResultsWith4DTensor) {
    // Create 4D test data (small dimensions for speed)
    std::array<size_t, 4> dims = {10, 8, 10, 8};
    // ... fill with known function ...

    // Fit without workspace (baseline)
    // ... existing fit code ...

    // Fit with workspace (new code)
    // ... parallel fit_axis with ThreadWorkspaceBuffer ...

    // Compare coefficients
    // EXPECT_NEAR for all coefficients
}
```

This test will be completed when BSplineNDSeparable is updated in Task 1.6.

**Verification:** `bazel test //tests:bspline_nd_workspace_integration_test`

---

### Task 1.6: Update BSplineNDSeparable::fit_axis to use workspace

**File:** `src/math/bspline_nd_separable.hpp`

**Implementation:**
Update the `fit_axis` template method to:
1. Create `ThreadWorkspaceBuffer` per thread (outside loop)
2. Create `BSplineCollocationWorkspace` once per thread
3. Use `fit_with_workspace` in the inner loop
4. Add critical section for statistics reduction

Follow the exact pattern from design document lines 563-676.

**Verification:** `bazel test //tests:bspline_nd_workspace_integration_test`

---

## Phase 2: American PDE Workspace

### Task 2.1: Create AmericanPDEWorkspace

**File:** `src/pde/core/american_pde_workspace.hpp` (NEW)

**Test (RED):**
```cpp
// tests/american_pde_workspace_test.cc
#include "src/pde/core/american_pde_workspace.hpp"
#include "src/support/thread_workspace.hpp"
#include <gtest/gtest.h>

using namespace mango;

TEST(AmericanPDEWorkspaceTest, RequiredBytesMatchesPDEWorkspace) {
    const size_t n = 101;

    size_t pde_doubles = PDEWorkspace::required_size(n);
    size_t american_bytes = AmericanPDEWorkspace::required_bytes(n);

    // Should be at least pde_doubles * sizeof(double)
    EXPECT_GE(american_bytes, pde_doubles * sizeof(double));
    // Should be 64-byte aligned
    EXPECT_EQ(american_bytes % 64, 0u);
}

TEST(AmericanPDEWorkspaceTest, FromBytesSuccess) {
    const size_t n = 101;
    ThreadWorkspaceBuffer buffer(AmericanPDEWorkspace::required_bytes(n));

    auto result = AmericanPDEWorkspace::from_bytes(buffer.bytes(), n);

    ASSERT_TRUE(result.has_value()) << result.error();
    auto& ws = result.value();

    EXPECT_EQ(ws.size(), n);
    EXPECT_EQ(ws.dx().size(), n - 1);
    EXPECT_EQ(ws.u_stage().size(), n);
    EXPECT_EQ(ws.rhs().size(), n);
}

TEST(AmericanPDEWorkspaceTest, BufferTooSmall) {
    const size_t n = 101;
    std::vector<std::byte> small(100);  // Way too small

    auto result = AmericanPDEWorkspace::from_bytes(std::span(small), n);

    EXPECT_FALSE(result.has_value());
}

TEST(AmericanPDEWorkspaceTest, GridSizeTooSmall) {
    ThreadWorkspaceBuffer buffer(1024);

    auto result = AmericanPDEWorkspace::from_bytes(buffer.bytes(), 1);

    EXPECT_FALSE(result.has_value());
}
```

**Implementation:**
Create `src/pde/core/american_pde_workspace.hpp` with the exact implementation from design document (lines 407-514).

**Verification:** `bazel test //tests:american_pde_workspace_test`

---

### Task 2.2: Add BUILD target for AmericanPDEWorkspace

**File:** `src/pde/core/BUILD.bazel`

Add:
```python
cc_library(
    name = "american_pde_workspace",
    hdrs = ["american_pde_workspace.hpp"],
    deps = [
        ":pde_workspace",
        "//src/support:lifetime",
    ],
    visibility = ["//visibility:public"],
)
```

**Verification:** `bazel build //src/pde/core:american_pde_workspace`

---

## Phase 3: American Batch Solver Refactor

### Task 3.1: Update american_option_batch.cpp to use ThreadWorkspaceBuffer

**File:** `src/option/american_option_batch.cpp`

**Test (RED):**
```cpp
// tests/american_option_batch_workspace_test.cc
#include "src/option/american_option_batch.hpp"
#include <gtest/gtest.h>

using namespace mango;

TEST(AmericanOptionBatchWorkspaceTest, BatchResultsUnchanged) {
    // Create batch of options
    std::vector<PricingParams> batch;
    for (int i = 0; i < 10; ++i) {
        batch.push_back(PricingParams{
            .spot = 100.0,
            .strike = 90.0 + i * 2.0,
            .maturity = 1.0,
            .volatility = 0.20,
            .rate = 0.05,
            .dividend_yield = 0.02,
            .type = OptionType::PUT
        });
    }

    // Solve batch
    auto results = solve_american_batch(batch);

    // Verify reasonable results
    for (size_t i = 0; i < results.size(); ++i) {
        ASSERT_TRUE(results[i].has_value()) << "Option " << i << " failed";
        EXPECT_GT(results[i]->price(), 0.0);
        EXPECT_LT(results[i]->delta(), 0.0);  // Put delta negative
    }
}
```

**Implementation:**
Update the parallel region in `american_option_batch.cpp` to use:
```cpp
MANGO_PRAGMA_PARALLEL
{
    ThreadWorkspaceBuffer buffer(AmericanPDEWorkspace::required_bytes(n_space));

    // Create workspace ONCE per thread
    auto ws = AmericanPDEWorkspace::from_bytes(buffer.bytes(), n_space).value();

    MANGO_PRAGMA_FOR_STATIC
    for (size_t i = 0; i < batch_size; ++i) {
        // Reuse ws - solver overwrites arrays each iteration
        solver.solve_with_workspace(options[i], ws);
    }
}
```

**Verification:** `bazel test //tests:american_option_batch_workspace_test`

---

## Verification

### Final Verification Checklist

After all tasks complete:

1. **All tests pass:** `bazel test //...`
2. **Examples compile:** `bazel build //examples/...`
3. **Benchmarks compile:** `bazel build //benchmarks/...`
4. **Python bindings compile:** `bazel build //python:mango_iv`
5. **No compiler warnings**

### Performance Benchmark

Create `benchmarks/bspline_workspace_benchmark.cc`:
```cpp
// Compare allocation counts before/after
// Measure time for 4D price table construction
```

Run: `bazel run //benchmarks:bspline_workspace_benchmark`

**Expected improvement:** 24,000 allocations → N (thread count) allocations

---

## Execution

To execute this plan, use:
```
/superpowers:execute-plan
```

Or for subagent-driven development:
```
Use superpowers:subagent-driven-development
```
