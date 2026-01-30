# Phase 1 Weeks 3-4: Multi-Dimensional Grids + Index Options Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement MultiGridBuffer for multi-dimensional price tables, fix dividend handling bugs in index option pricing, and create CPU-only WorkspaceStorage specialization.

**Architecture:** Multi-dimensional grid infrastructure using axis-based buffers, proper 5D price table precomputation with dividend dimension, and CPU-optimized workspace storage. This extends the 1D grid system from Weeks 1-2 to support N-dimensional price tables while maintaining the same span-based, cache-aware design.

**Tech Stack:** C++20, std::unordered_map for axis storage, std::span views, GoogleTest

---

## Task 1: Implement MultiGridBuffer with Axis Management

**Files:**
- Create: `src/cpp/multigrid.hpp`
- Test: `tests/multigrid_test.cc`
- Modify: `src/cpp/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write failing test for 2D grid (moneyness × maturity)**

Create `tests/multigrid_test.cc`:

```cpp
#include "src/cpp/multigrid.hpp"
#include "src/cpp/grid.hpp"
#include <gtest/gtest.h>

TEST(MultiGridBufferTest, TwoAxisCreation) {
    mango::MultiGridBuffer mgrid;

    // Add moneyness axis: log-spaced [0.7, 1.3] with 10 points
    auto m_spec = mango::GridSpec<>::log_spaced(0.7, 1.3, 10);
    mgrid.add_axis(mango::GridAxis::Moneyness, m_spec);

    // Add maturity axis: linear [0.027, 2.0] with 20 points
    auto tau_spec = mango::GridSpec<>::uniform(0.027, 2.0, 20);
    mgrid.add_axis(mango::GridAxis::Maturity, tau_spec);

    // Verify axes were added
    EXPECT_TRUE(mgrid.has_axis(mango::GridAxis::Moneyness));
    EXPECT_TRUE(mgrid.has_axis(mango::GridAxis::Maturity));
    EXPECT_FALSE(mgrid.has_axis(mango::GridAxis::Volatility));

    // Verify axis sizes
    EXPECT_EQ(mgrid.axis_size(mango::GridAxis::Moneyness), 10);
    EXPECT_EQ(mgrid.axis_size(mango::GridAxis::Maturity), 20);

    // Verify total grid points
    EXPECT_EQ(mgrid.total_points(), 200);  // 10 × 20
}

TEST(MultiGridBufferTest, AccessAxisData) {
    mango::MultiGridBuffer mgrid;

    auto m_spec = mango::GridSpec<>::uniform(0.8, 1.2, 5);
    mgrid.add_axis(mango::GridAxis::Moneyness, m_spec);

    // Get view of moneyness axis
    auto m_view = mgrid.axis_view(mango::GridAxis::Moneyness);

    // Check endpoints
    EXPECT_DOUBLE_EQ(m_view[0], 0.8);
    EXPECT_DOUBLE_EQ(m_view[4], 1.2);

    // Check uniform spacing
    double expected_spacing = (1.2 - 0.8) / 4.0;  // 0.1
    for (size_t i = 0; i < 4; ++i) {
        double spacing = m_view[i+1] - m_view[i];
        EXPECT_NEAR(spacing, expected_spacing, 1e-10);
    }
}
```

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "multigrid_test",
    srcs = ["multigrid_test.cc"],
    deps = [
        "//src/cpp:multigrid",
        "//src/cpp:grid",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
    copts = ["-std=c++20", "-Wall", "-Wextra"],
)
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:multigrid_test --test_output=all
```

Expected: FAIL with "multigrid.hpp: No such file or directory"

**Step 3: Implement MultiGridBuffer**

Create `src/cpp/multigrid.hpp`:

```cpp
#pragma once

#include "grid.hpp"
#include <unordered_map>
#include <stdexcept>
#include <numeric>

namespace mango {

/// Grid axes for multi-dimensional grids
enum class GridAxis {
    Space,      // PDE solver spatial dimension
    Time,       // PDE solver time dimension
    Moneyness,  // Price table: S/K ratio
    Maturity,   // Price table: time to maturity
    Volatility, // Price table: implied volatility
    Rate,       // Price table: risk-free rate
    Dividend    // Price table: dividend yield
};

/// Multi-dimensional grid container
///
/// Manages multiple GridBuffer objects indexed by GridAxis.
/// Each axis is independent and can have different spacing (uniform, log, sinh).
/// Total grid points = product of all axis sizes.
///
/// Example: 4D price table (moneyness × maturity × volatility × rate)
/// - 50 moneyness points × 30 maturity × 20 volatility × 10 rate = 300,000 points
class MultiGridBuffer {
public:
    /// Add an axis to the multi-dimensional grid
    ///
    /// @param axis Axis identifier (e.g., GridAxis::Moneyness)
    /// @param spec Grid specification for this axis
    ///
    /// Throws std::runtime_error if axis already exists.
    void add_axis(GridAxis axis, const GridSpec<>& spec) {
        if (buffers_.contains(axis)) {
            throw std::runtime_error("Axis already exists in MultiGridBuffer");
        }
        buffers_[axis] = spec.generate();
    }

    /// Check if an axis exists
    bool has_axis(GridAxis axis) const {
        return buffers_.contains(axis);
    }

    /// Get size of a specific axis
    ///
    /// @param axis Axis identifier
    /// @return Number of points along this axis
    ///
    /// Throws std::out_of_range if axis does not exist.
    size_t axis_size(GridAxis axis) const {
        return buffers_.at(axis).size();
    }

    /// Get total number of grid points (product of all axis sizes)
    size_t total_points() const {
        if (buffers_.empty()) return 0;

        return std::accumulate(
            buffers_.begin(), buffers_.end(), size_t{1},
            [](size_t product, const auto& pair) {
                return product * pair.second.size();
            }
        );
    }

    /// Get view of axis data
    ///
    /// @param axis Axis identifier
    /// @return Span view of grid points for this axis
    ///
    /// Throws std::out_of_range if axis does not exist.
    std::span<const double> axis_view(GridAxis axis) const {
        return buffers_.at(axis).span();
    }

    /// Get number of axes
    size_t n_axes() const {
        return buffers_.size();
    }

private:
    std::unordered_map<GridAxis, GridBuffer<>> buffers_;
};

}  // namespace mango
```

Add to `src/cpp/BUILD.bazel`:

```python
cc_library(
    name = "multigrid",
    hdrs = ["multigrid.hpp"],
    deps = [":grid"],
    copts = ["-std=c++20", "-Wall", "-Wextra"],
    visibility = ["//visibility:public"],
)
```

**Step 4: Run test to verify it passes**

```bash
bazel test //tests:multigrid_test --test_output=all
```

Expected: PASS (2/2 tests)

**Step 5: Commit**

```bash
git add src/cpp/multigrid.hpp src/cpp/BUILD.bazel tests/multigrid_test.cc tests/BUILD.bazel
git commit -m "feat(multigrid): add MultiGridBuffer for N-dimensional grids

- Axis-based grid management using unordered_map
- Support for independent axis spacing (uniform, log, sinh)
- Total grid points computed as product of axis sizes
- Span-based views for zero-copy axis access

Tests: 2 passing"
```

---

## Task 2: Add 5D Price Table Test with Dividend Dimension

**Files:**
- Modify: `tests/multigrid_test.cc`

**Step 1: Write failing test for 5D price table**

Add to `tests/multigrid_test.cc`:

```cpp
TEST(MultiGridBufferTest, FiveDimensionalPriceTable) {
    mango::MultiGridBuffer mgrid;

    // 5D price table: moneyness × maturity × volatility × rate × dividend
    mgrid.add_axis(mango::GridAxis::Moneyness,  mango::GridSpec<>::log_spaced(0.7, 1.3, 50));
    mgrid.add_axis(mango::GridAxis::Maturity,   mango::GridSpec<>::uniform(0.027, 2.0, 30));
    mgrid.add_axis(mango::GridAxis::Volatility, mango::GridSpec<>::uniform(0.10, 0.80, 20));
    mgrid.add_axis(mango::GridAxis::Rate,       mango::GridSpec<>::uniform(0.0, 0.10, 10));
    mgrid.add_axis(mango::GridAxis::Dividend,   mango::GridSpec<>::uniform(0.0, 0.05, 5));

    // Verify all axes present
    EXPECT_EQ(mgrid.n_axes(), 5);
    EXPECT_TRUE(mgrid.has_axis(mango::GridAxis::Dividend));

    // Verify total points
    size_t expected_total = 50 * 30 * 20 * 10 * 5;  // 1,500,000 points
    EXPECT_EQ(mgrid.total_points(), expected_total);

    // Verify dividend axis spacing
    auto div_view = mgrid.axis_view(mango::GridAxis::Dividend);
    EXPECT_EQ(div_view.size(), 5);
    EXPECT_DOUBLE_EQ(div_view[0], 0.0);
    EXPECT_DOUBLE_EQ(div_view[4], 0.05);
}
```

**Step 2: Run test to verify it passes**

```bash
bazel test //tests:multigrid_test::FiveDimensionalPriceTable --test_output=all
```

Expected: PASS (implementation already supports N dimensions)

**Step 3: Commit**

```bash
git add tests/multigrid_test.cc
git commit -m "test(multigrid): add 5D price table test with dividend

Verifies MultiGridBuffer handles 5-dimensional grids:
- 1.5M grid points (50×30×20×10×5)
- Dividend axis included for index options
- All axes independently configurable

Tests: 3 passing"
```

---

## Task 3: Fix IndexBlackScholesOperator to Use Pre-Computed dx

**Files:**
- Modify: `src/cpp/spatial_operators.hpp`
- Modify: `tests/spatial_operators_test.cc`

**Step 1: Write failing test for operator with pre-computed dx**

Add to `tests/spatial_operators_test.cc`:

```cpp
TEST(BlackScholesOperatorTest, OperatorUsesPrecomputedDx) {
    // Create non-uniform grid to ensure dx matters
    auto spec = mango::GridSpec<>::sinh_spaced(50.0, 150.0, 21, 100.0, 1.5);
    auto grid = spec.generate();

    // Create workspace with pre-computed dx
    mango::WorkspaceStorage workspace(grid.size(), grid.span());

    // Initialize test function u(S) = S
    auto u = workspace.u_current();
    for (size_t i = 0; i < grid.size(); ++i) {
        u[i] = grid[i];
    }

    // Apply index operator WITH dx parameter
    mango::IndexBlackScholesOperator op(0.05, 0.2, 0.03);
    auto Lu = workspace.lu();

    // NEW SIGNATURE: pass pre-computed dx
    op.apply(0.0, grid.span(), u, Lu, workspace.dx());

    // For u(S) = S: L(u) = -q*S
    double S_mid = grid[10];  // Middle of grid
    double expected = -0.03 * S_mid;
    EXPECT_NEAR(Lu[10], expected, 0.01);
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:spatial_operators_test::OperatorUsesPrecomputedDx --test_output=all
```

Expected: FAIL - "no matching function for call to 'apply'" (missing dx parameter)

**Step 3: Update operator signature to accept dx**

Modify `src/cpp/spatial_operators.hpp`:

```cpp
// Update EquityBlackScholesOperator
class EquityBlackScholesOperator {
public:
    // ... existing constructor ...

    /// Apply Black-Scholes operator: L(V) = 0.5σ²S²∂²V/∂S² + rS∂V/∂S - rV
    ///
    /// @param t Current time
    /// @param S Stock price grid
    /// @param u Solution values
    /// @param Lu Output: operator applied to u
    /// @param dx Pre-computed grid spacing (size n-1)
    void apply(double t, std::span<const double> S,
               std::span<const double> u, std::span<double> Lu,
               std::span<const double> dx) const {
        const size_t n = S.size();

        // Boundary points set to zero (handled by BC application)
        Lu[0] = Lu[n-1] = 0.0;

        // Interior points: use pre-computed dx
        for (size_t i = 1; i < n - 1; ++i) {
            const double S_i = S[i];

            // Use PRE-COMPUTED grid spacing
            const double dx_left = dx[i-1];   // S[i] - S[i-1]
            const double dx_right = dx[i];     // S[i+1] - S[i]
            const double dx_center = 0.5 * (dx_left + dx_right);

            // First derivative: centered difference
            const double du_dS = (u[i+1] - u[i-1]) / (dx_left + dx_right);

            // Second derivative: non-uniform grid formula
            const double d2u = (u[i+1] - u[i]) / dx_right - (u[i] - u[i-1]) / dx_left;
            const double d2u_dS2 = d2u / dx_center;

            // Black-Scholes operator for equity options
            Lu[i] = sigma_sq_half_ * S_i * S_i * d2u_dS2
                  + r_ * S_i * du_dS
                  - r_ * u[i];
        }
    }

private:
    double r_;
    double sigma_sq_half_;
};

// Update IndexBlackScholesOperator similarly
class IndexBlackScholesOperator {
public:
    // ... existing constructor ...

    /// Apply Black-Scholes operator for index options: L(V) = 0.5σ²S²∂²V/∂S² + (r-q)S∂V/∂S - rV
    ///
    /// @param t Current time
    /// @param S Stock price grid
    /// @param u Solution values
    /// @param Lu Output: operator applied to u
    /// @param dx Pre-computed grid spacing (size n-1)
    void apply(double t, std::span<const double> S,
               std::span<const double> u, std::span<double> Lu,
               std::span<const double> dx) const {
        const size_t n = S.size();

        Lu[0] = Lu[n-1] = 0.0;

        for (size_t i = 1; i < n - 1; ++i) {
            const double S_i = S[i];

            // Use PRE-COMPUTED grid spacing
            const double dx_left = dx[i-1];
            const double dx_right = dx[i];
            const double dx_center = 0.5 * (dx_left + dx_right);

            const double du_dS = (u[i+1] - u[i-1]) / (dx_left + dx_right);

            const double d2u = (u[i+1] - u[i]) / dx_right - (u[i] - u[i-1]) / dx_left;
            const double d2u_dS2 = d2u / dx_center;

            // Black-Scholes operator for index options (includes dividend yield q)
            Lu[i] = sigma_sq_half_ * S_i * S_i * d2u_dS2
                  + (r_ - q_) * S_i * du_dS  // Drift with dividend
                  - r_ * u[i];
        }
    }

private:
    double r_;
    double q_;
    double sigma_sq_half_;
};
```

**Step 4: Update existing tests to use new signature**

Modify existing tests in `tests/spatial_operators_test.cc`:

```cpp
TEST(BlackScholesOperatorTest, EquityOperatorBasic) {
    auto spec = mango::GridSpec<>::uniform(50.0, 150.0, 21);
    auto grid = spec.generate();
    mango::WorkspaceStorage workspace(grid.size(), grid.span());

    // ... existing setup ...

    // UPDATE: pass dx parameter
    mango::EquityBlackScholesOperator op(0.05, 0.2);
    op.apply(0.0, grid.span(), u, Lu, workspace.dx());

    // ... existing assertions ...
}

// Repeat for all other tests...
```

**Step 5: Run tests to verify they pass**

```bash
bazel test //tests:spatial_operators_test --test_output=all
```

Expected: PASS (5/5 tests, including new test)

**Step 6: Commit**

```bash
git add src/cpp/spatial_operators.hpp tests/spatial_operators_test.cc
git commit -m "fix(operators): use pre-computed dx from WorkspaceStorage

- Updated EquityBlackScholesOperator::apply() to accept dx span
- Updated IndexBlackScholesOperator::apply() to accept dx span
- Eliminates redundant S[i+1] - S[i] computation on every call
- All existing tests updated to new signature

Performance: Saves O(n) floating-point ops per operator application

Tests: 5 passing"
```

---

## Task 4: Add WorkspaceStorage CPU Specialization Documentation

**Files:**
- Modify: `src/cpp/workspace.hpp`

**Step 1: Add CPU-only specialization comments**

Modify `src/cpp/workspace.hpp`:

```cpp
#pragma once

#include "grid.hpp"
#include "cache_config.hpp"
#include <vector>
#include <span>
#include <cstddef>

namespace mango {

/// Workspace storage for PDE solver arrays
///
/// **CPU-only implementation** - SYCL GPU specialization deferred to v2.1.
///
/// Manages all solver state in a single contiguous buffer for cache efficiency.
/// Arrays: u_current, u_next, u_stage, rhs, Lu (5n doubles total).
///
/// **Pre-computed dx array** - Grid spacing computed once during construction
/// to avoid redundant S[i+1] - S[i] calculations in stencil operations.
///
/// **Cache-blocking** - Adaptive strategy based on grid size:
/// - n < 5000: Single block (no blocking overhead)
/// - n ≥ 5000: L1-blocked (~1000 points per block, ~32 KB working set)
///
/// Future GPU version (v2.1) will use SYCL unified shared memory (USM)
/// with explicit device allocation and host-device synchronization.
class WorkspaceStorage {
public:
    /// Construct workspace for n grid points
    ///
    /// @param n Number of grid points
    /// @param grid Grid coordinates for pre-computing dx
    ///
    /// Allocates 5n doubles (u_current, u_next, u_stage, rhs, Lu)
    /// plus (n-1) doubles for pre-computed dx array.
    explicit WorkspaceStorage(size_t n, std::span<const double> grid)
        : buffer_(5 * n)
        , cache_config_(CacheBlockConfig::adaptive(n))
        , dx_(n - 1)
    {
        // Pre-compute grid spacing once during initialization
        // CRITICAL: Avoids out-of-bounds access when processing cache blocks
        for (size_t i = 0; i < n - 1; ++i) {
            dx_[i] = grid[i + 1] - grid[i];
        }

        // ... rest of existing implementation ...
    }

    // ... rest of existing methods ...

private:
    std::vector<double> buffer_;     // Single allocation for all arrays (CPU memory)
    CacheBlockConfig cache_config_;  // Cache-blocking configuration (CPU-only)
    std::vector<double> dx_;         // Pre-computed grid spacing

    // ... rest of existing members ...
};

}  // namespace mango
```

**Step 2: Run tests to verify documentation doesn't break anything**

```bash
bazel test //tests:workspace_test --test_output=all
```

Expected: PASS (5/5 tests)

**Step 3: Commit**

```bash
git add src/cpp/workspace.hpp
git commit -m "docs(workspace): clarify CPU-only implementation

- Added documentation stating this is CPU-only (SYCL deferred to v2.1)
- Explained single contiguous buffer allocation strategy
- Clarified pre-computed dx purpose and cache-blocking rationale
- Future GPU version will use SYCL USM for device allocation

No functional changes - documentation only"
```

---

## Task 5: Integration Test - 5D Price Table Grid Setup

**Files:**
- Create: `tests/integration_5d_price_table_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write integration test for 5D grid with all components**

Create `tests/integration_5d_price_table_test.cc`:

```cpp
#include "src/cpp/multigrid.hpp"
#include "src/cpp/workspace.hpp"
#include "src/cpp/spatial_operators.hpp"
#include "src/cpp/grid.hpp"
#include <gtest/gtest.h>

/// Integration test: 5D price table grid setup
///
/// Verifies MultiGridBuffer works with spatial operators and workspace storage.
/// This simulates the grid setup phase of price table precomputation.
TEST(Integration5DPriceTableTest, GridSetupWithDividendDimension) {
    // Step 1: Create 5D price table grid
    mango::MultiGridBuffer price_table_grid;

    price_table_grid.add_axis(mango::GridAxis::Moneyness,
                              mango::GridSpec<>::log_spaced(0.7, 1.3, 10));
    price_table_grid.add_axis(mango::GridAxis::Maturity,
                              mango::GridSpec<>::uniform(0.027, 2.0, 8));
    price_table_grid.add_axis(mango::GridAxis::Volatility,
                              mango::GridSpec<>::uniform(0.10, 0.50, 5));
    price_table_grid.add_axis(mango::GridAxis::Rate,
                              mango::GridSpec<>::uniform(0.0, 0.10, 4));
    price_table_grid.add_axis(mango::GridAxis::Dividend,
                              mango::GridSpec<>::uniform(0.0, 0.05, 3));

    // Verify grid dimensions
    EXPECT_EQ(price_table_grid.n_axes(), 5);
    size_t expected_combinations = 10 * 8 * 5 * 4 * 3;  // 4,800 parameter combinations
    EXPECT_EQ(price_table_grid.total_points(), expected_combinations);

    // Step 2: Extract dividend axis and verify spacing
    auto dividend_axis = price_table_grid.axis_view(mango::GridAxis::Dividend);
    EXPECT_EQ(dividend_axis.size(), 3);
    EXPECT_DOUBLE_EQ(dividend_axis[0], 0.0);    // No dividend
    EXPECT_DOUBLE_EQ(dividend_axis[1], 0.025);  // 2.5% yield
    EXPECT_DOUBLE_EQ(dividend_axis[2], 0.05);   // 5% yield

    // Step 3: Create PDE solver spatial grid (for a single parameter combination)
    // This would be created once per parameter combination during precompute
    auto pde_grid_spec = mango::GridSpec<>::log_spaced(50.0, 150.0, 101);
    auto pde_grid = pde_grid_spec.generate();

    // Step 4: Create workspace for this PDE solve
    mango::WorkspaceStorage workspace(pde_grid.size(), pde_grid.span());

    // Verify workspace has pre-computed dx
    EXPECT_EQ(workspace.dx().size(), 100);  // n-1 spacing values

    // Step 5: Create index operator with dividend from price table
    double dividend_yield = dividend_axis[1];  // Use q=0.025 from table
    mango::IndexBlackScholesOperator op(0.05, 0.3, dividend_yield);

    // Step 6: Initialize test solution and apply operator
    auto u = workspace.u_current();
    for (size_t i = 0; i < pde_grid.size(); ++i) {
        u[i] = pde_grid[i];  // u(S) = S
    }

    auto Lu = workspace.lu();
    op.apply(0.0, pde_grid.span(), u, Lu, workspace.dx());

    // Step 7: Verify operator used dividend correctly: L(S) = -q*S
    double S_mid = pde_grid[50];
    double expected_Lu = -dividend_yield * S_mid;
    EXPECT_NEAR(Lu[50], expected_Lu, 0.01);

    // This confirms the full pipeline:
    // 5D grid → dividend axis → index operator → workspace → dx → result
}

/// Test: Verify dividend dimension affects operator output
TEST(Integration5DPriceTableTest, DividendAffectsPriceCalculation) {
    // Create small 2D slice: dividend × rate
    mango::MultiGridBuffer grid;
    grid.add_axis(mango::GridAxis::Dividend, mango::GridSpec<>::uniform(0.0, 0.04, 3));
    grid.add_axis(mango::GridAxis::Rate,     mango::GridSpec<>::uniform(0.02, 0.06, 3));

    // PDE solver grid (same for all parameter combinations)
    auto pde_spec = mango::GridSpec<>::uniform(80.0, 120.0, 41);
    auto pde_grid = pde_spec.generate();
    mango::WorkspaceStorage workspace(pde_grid.size(), pde_grid.span());

    auto div_axis = grid.axis_view(mango::GridAxis::Dividend);
    auto rate_axis = grid.axis_view(mango::GridAxis::Rate);

    // Test two parameter combinations with different dividends
    double r = rate_axis[1];  // r = 0.04
    double q1 = div_axis[0];  // q = 0.0
    double q2 = div_axis[2];  // q = 0.04

    // Create operators
    mango::IndexBlackScholesOperator op1(r, 0.2, q1);
    mango::IndexBlackScholesOperator op2(r, 0.2, q2);

    // Initialize u(S) = S
    auto u = workspace.u_current();
    for (size_t i = 0; i < pde_grid.size(); ++i) {
        u[i] = pde_grid[i];
    }

    // Apply both operators
    auto Lu1 = workspace.lu();
    op1.apply(0.0, pde_grid.span(), u, Lu1, workspace.dx());

    std::vector<double> Lu2_buffer(pde_grid.size());
    std::span<double> Lu2(Lu2_buffer);
    op2.apply(0.0, pde_grid.span(), u, Lu2, workspace.dx());

    // Verify difference is due to dividend: Lu1 - Lu2 = (q2 - q1)*S
    double S_mid = pde_grid[20];
    double expected_diff = (q2 - q1) * S_mid;  // 0.04 * S
    double actual_diff = Lu1[20] - Lu2[20];

    EXPECT_NEAR(actual_diff, expected_diff, 0.01);

    // This confirms dividend dimension properly affects PDE operator
}
```

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "integration_5d_price_table_test",
    srcs = ["integration_5d_price_table_test.cc"],
    deps = [
        "//src/cpp:multigrid",
        "//src/cpp:workspace",
        "//src/cpp:spatial_operators",
        "//src/cpp:grid",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
    copts = ["-std=c++20", "-Wall", "-Wextra"],
)
```

**Step 2: Run integration test**

```bash
bazel test //tests:integration_5d_price_table_test --test_output=all
```

Expected: PASS (2/2 tests)

**Step 3: Commit**

```bash
git add tests/integration_5d_price_table_test.cc tests/BUILD.bazel
git commit -m "test(integration): add 5D price table grid setup tests

Integration tests verify complete pipeline:
- MultiGridBuffer with 5 dimensions (including dividend)
- Extract dividend values from grid axis
- Create IndexBlackScholesOperator with table dividend
- WorkspaceStorage with pre-computed dx
- Operator application uses dx correctly
- Dividend dimension affects operator output

Tests demonstrate full workflow for price table precomputation.

Tests: 2 passing"
```

---

## Summary

This plan implements Phase 1 Weeks 3-4 deliverables:

1. ✅ **MultiGridBuffer** - Multi-dimensional grid infrastructure
2. ✅ **5D Price Table Support** - Including dividend dimension
3. ✅ **Fixed Bug** - IndexBlackScholesOperator uses pre-computed dx
4. ✅ **CPU-Only WorkspaceStorage** - Documented SYCL deferral to v2.1
5. ✅ **Integration Tests** - Verified complete 5D → dividend → operator → workspace pipeline

**Total: 5 tasks, ~15 test cases**

**Next Phase:** Weeks 5-6 will implement TR-BDF2 solver with cache-blocked stencils.
