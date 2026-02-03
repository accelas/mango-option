# Price Table Arrow IPC Persistence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable zero-copy mmap loading of pre-computed option price tables using Apache Arrow IPC format for sub-millisecond startup in production trading systems.

**Architecture:** Refactor `BSpline4D` and `PriceTableSurface` to use arena-allocated workspace pattern (similar to `AmericanSolverWorkspace`). Add Arrow IPC save/load with embedded schema validation, checksums, and 64-byte alignment for AVX-512 SIMD.

**Tech Stack:** C++23, Apache Arrow C++ (IPC format), std::span for zero-copy views, CRC64 for data integrity, mmap for file loading.

---

## Phase 1: Workspace Foundation

### Task 1: Create PriceTableWorkspace with Aligned Arena

**Files:**
- Create: `src/option/price_table_workspace.hpp`
- Create: `src/option/price_table_workspace.cpp`
- Test: `tests/price_table_workspace_test.cc`

**Step 1: Write the failing test**

Create `tests/price_table_workspace_test.cc`:

```cpp
#include "mango/option/price_table_workspace.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(PriceTableWorkspace, ConstructsFromGridData) {
    std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1};
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 1.0);

    auto ws_result = mango::PriceTableWorkspace::create(
        m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02);

    ASSERT_TRUE(ws_result.has_value());
    auto& ws = ws_result.value();

    EXPECT_EQ(ws.moneyness().size(), 4);
    EXPECT_EQ(ws.maturity().size(), 4);
    EXPECT_EQ(ws.coefficients().size(), 256);
    EXPECT_DOUBLE_EQ(ws.K_ref(), 100.0);
}

TEST(PriceTableWorkspace, RejectsInsufficientGridPoints) {
    std::vector<double> m_grid = {0.9, 1.0, 1.1};  // Only 3 points
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(3 * 4 * 4 * 4, 1.0);

    auto ws_result = mango::PriceTableWorkspace::create(
        m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02);

    EXPECT_FALSE(ws_result.has_value());
    EXPECT_EQ(ws_result.error(), "Moneyness grid must have >= 4 points");
}

TEST(PriceTableWorkspace, ValidatesArenaAlignment) {
    std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1};
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 1.0);

    auto ws_result = mango::PriceTableWorkspace::create(
        m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02);

    ASSERT_TRUE(ws_result.has_value());
    auto& ws = ws_result.value();

    // Check 64-byte alignment for SIMD
    auto addr = reinterpret_cast<std::uintptr_t>(ws.moneyness().data());
    EXPECT_EQ(addr % 64, 0) << "Moneyness grid not 64-byte aligned";
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:price_table_workspace_test --test_output=all
```

Expected: FAIL with "No such file: src/option/price_table_workspace.hpp"

**Step 3: Write minimal implementation header**

Create `src/option/price_table_workspace.hpp`:

```cpp
#pragma once

#include "mango/support/expected.hpp"
#include "mango/interpolation/bspline_utils.hpp"
#include <vector>
#include <span>
#include <string>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace mango {

/// Workspace holding all data for PriceTableSurface in single contiguous allocation
///
/// Enables zero-copy mmap loading from Arrow IPC files. All numeric data is
/// 64-byte aligned for AVX-512 SIMD operations.
///
/// Memory layout (all contiguous):
///   [moneyness grid][maturity grid][volatility grid][rate grid]
///   [knots_m][knots_tau][knots_sigma][knots_r]
///   [coefficients]
///   [metadata: K_ref, dividend_yield]
///
/// Example:
///   auto ws = PriceTableWorkspace::create(m, tau, sigma, r, coeffs, K_ref, q);
///   BSpline4D spline(ws.value());
class PriceTableWorkspace {
public:
    /// Factory method with validation
    ///
    /// @param m_grid Moneyness grid (sorted ascending, >= 4 points)
    /// @param tau_grid Maturity grid (years, sorted ascending, >= 4 points)
    /// @param sigma_grid Volatility grid (sorted ascending, >= 4 points)
    /// @param r_grid Rate grid (sorted ascending, >= 4 points)
    /// @param coefficients B-spline coefficients (size = n_m * n_tau * n_sigma * n_r)
    /// @param K_ref Reference strike price
    /// @param dividend_yield Continuous dividend yield
    /// @return Expected workspace or error message
    static expected<PriceTableWorkspace, std::string> create(
        const std::vector<double>& m_grid,
        const std::vector<double>& tau_grid,
        const std::vector<double>& sigma_grid,
        const std::vector<double>& r_grid,
        const std::vector<double>& coefficients,
        double K_ref,
        double dividend_yield);

    /// Grid accessors (zero-copy spans into arena)
    std::span<const double> moneyness() const { return moneyness_; }
    std::span<const double> maturity() const { return maturity_; }
    std::span<const double> volatility() const { return volatility_; }
    std::span<const double> rate() const { return rate_; }

    /// Knot vector accessors (precomputed clamped cubic knots)
    std::span<const double> knots_moneyness() const { return knots_m_; }
    std::span<const double> knots_maturity() const { return knots_tau_; }
    std::span<const double> knots_volatility() const { return knots_sigma_; }
    std::span<const double> knots_rate() const { return knots_r_; }

    /// Coefficient accessor (4D tensor in row-major layout)
    std::span<const double> coefficients() const { return coefficients_; }

    /// Metadata accessors
    double K_ref() const { return K_ref_; }
    double dividend_yield() const { return dividend_yield_; }

    /// Grid dimensions
    std::tuple<size_t, size_t, size_t, size_t> dimensions() const {
        return {moneyness_.size(), maturity_.size(),
                volatility_.size(), rate_.size()};
    }

    /// Move-only semantics (no copies of large arena)
    PriceTableWorkspace(const PriceTableWorkspace&) = delete;
    PriceTableWorkspace& operator=(const PriceTableWorkspace&) = delete;
    PriceTableWorkspace(PriceTableWorkspace&&) noexcept = default;
    PriceTableWorkspace& operator=(PriceTableWorkspace&&) noexcept = default;

private:
    PriceTableWorkspace() = default;

    /// Allocate aligned arena and set up spans
    static expected<PriceTableWorkspace, std::string> allocate_and_initialize(
        const std::vector<double>& m_grid,
        const std::vector<double>& tau_grid,
        const std::vector<double>& sigma_grid,
        const std::vector<double>& r_grid,
        const std::vector<double>& coefficients,
        double K_ref,
        double dividend_yield);

    /// Validate grids before allocation
    static expected<void, std::string> validate_inputs(
        const std::vector<double>& m_grid,
        const std::vector<double>& tau_grid,
        const std::vector<double>& sigma_grid,
        const std::vector<double>& r_grid,
        const std::vector<double>& coefficients);

    // Single contiguous allocation (64-byte aligned)
    std::vector<double> arena_;

    // Views into arena (no ownership)
    std::span<const double> moneyness_;
    std::span<const double> maturity_;
    std::span<const double> volatility_;
    std::span<const double> rate_;

    std::span<const double> knots_m_;
    std::span<const double> knots_tau_;
    std::span<const double> knots_sigma_;
    std::span<const double> knots_r_;

    std::span<const double> coefficients_;

    // Scalar metadata
    double K_ref_ = 0.0;
    double dividend_yield_ = 0.0;
};

}  // namespace mango
```

**Step 4: Write implementation**

Create `src/option/price_table_workspace.cpp`:

```cpp
#include "mango/option/price_table_workspace.hpp"
#include <algorithm>
#include <numeric>
#include <cstring>

namespace mango {

expected<void, std::string> PriceTableWorkspace::validate_inputs(
    const std::vector<double>& m_grid,
    const std::vector<double>& tau_grid,
    const std::vector<double>& sigma_grid,
    const std::vector<double>& r_grid,
    const std::vector<double>& coefficients)
{
    // Validate grid sizes
    if (m_grid.size() < 4) {
        return unexpected("Moneyness grid must have >= 4 points");
    }
    if (tau_grid.size() < 4) {
        return unexpected("Maturity grid must have >= 4 points");
    }
    if (sigma_grid.size() < 4) {
        return unexpected("Volatility grid must have >= 4 points");
    }
    if (r_grid.size() < 4) {
        return unexpected("Rate grid must have >= 4 points");
    }

    // Validate coefficient size
    size_t expected_size = m_grid.size() * tau_grid.size() *
                          sigma_grid.size() * r_grid.size();
    if (coefficients.size() != expected_size) {
        return unexpected("Coefficient size mismatch: expected " +
                         std::to_string(expected_size) + ", got " +
                         std::to_string(coefficients.size()));
    }

    // Validate monotonicity
    auto is_sorted = [](const std::vector<double>& v) {
        return std::is_sorted(v.begin(), v.end());
    };

    if (!is_sorted(m_grid)) {
        return unexpected("Moneyness grid must be sorted ascending");
    }
    if (!is_sorted(tau_grid)) {
        return unexpected("Maturity grid must be sorted ascending");
    }
    if (!is_sorted(sigma_grid)) {
        return unexpected("Volatility grid must be sorted ascending");
    }
    if (!is_sorted(r_grid)) {
        return unexpected("Rate grid must be sorted ascending");
    }

    return {};
}

expected<PriceTableWorkspace, std::string> PriceTableWorkspace::allocate_and_initialize(
    const std::vector<double>& m_grid,
    const std::vector<double>& tau_grid,
    const std::vector<double>& sigma_grid,
    const std::vector<double>& r_grid,
    const std::vector<double>& coefficients,
    double K_ref,
    double dividend_yield)
{
    PriceTableWorkspace ws;

    // Compute knot vectors (clamped cubic B-spline)
    auto knots_m = clamped_knots_cubic(m_grid);
    auto knots_tau = clamped_knots_cubic(tau_grid);
    auto knots_sigma = clamped_knots_cubic(sigma_grid);
    auto knots_r = clamped_knots_cubic(r_grid);

    // Calculate total arena size
    size_t total_size = m_grid.size() + tau_grid.size() +
                       sigma_grid.size() + r_grid.size() +
                       knots_m.size() + knots_tau.size() +
                       knots_sigma.size() + knots_r.size() +
                       coefficients.size();

    // Allocate with 64-byte alignment for AVX-512
    // Use over-allocation to ensure alignment
    ws.arena_.resize(total_size + 8);  // +8 for alignment padding

    // Find 64-byte aligned start within arena
    auto arena_ptr = reinterpret_cast<std::uintptr_t>(ws.arena_.data());
    auto aligned_offset = (64 - (arena_ptr % 64)) % 64;
    double* aligned_start = ws.arena_.data() + aligned_offset / sizeof(double);

    // Copy data into arena
    double* ptr = aligned_start;

    std::memcpy(ptr, m_grid.data(), m_grid.size() * sizeof(double));
    ws.moneyness_ = std::span<const double>(ptr, m_grid.size());
    ptr += m_grid.size();

    std::memcpy(ptr, tau_grid.data(), tau_grid.size() * sizeof(double));
    ws.maturity_ = std::span<const double>(ptr, tau_grid.size());
    ptr += tau_grid.size();

    std::memcpy(ptr, sigma_grid.data(), sigma_grid.size() * sizeof(double));
    ws.volatility_ = std::span<const double>(ptr, sigma_grid.size());
    ptr += sigma_grid.size();

    std::memcpy(ptr, r_grid.data(), r_grid.size() * sizeof(double));
    ws.rate_ = std::span<const double>(ptr, r_grid.size());
    ptr += r_grid.size();

    std::memcpy(ptr, knots_m.data(), knots_m.size() * sizeof(double));
    ws.knots_m_ = std::span<const double>(ptr, knots_m.size());
    ptr += knots_m.size();

    std::memcpy(ptr, knots_tau.data(), knots_tau.size() * sizeof(double));
    ws.knots_tau_ = std::span<const double>(ptr, knots_tau.size());
    ptr += knots_tau.size();

    std::memcpy(ptr, knots_sigma.data(), knots_sigma.size() * sizeof(double));
    ws.knots_sigma_ = std::span<const double>(ptr, knots_sigma.size());
    ptr += knots_sigma.size();

    std::memcpy(ptr, knots_r.data(), knots_r.size() * sizeof(double));
    ws.knots_r_ = std::span<const double>(ptr, knots_r.size());
    ptr += knots_r.size();

    std::memcpy(ptr, coefficients.data(), coefficients.size() * sizeof(double));
    ws.coefficients_ = std::span<const double>(ptr, coefficients.size());

    ws.K_ref_ = K_ref;
    ws.dividend_yield_ = dividend_yield;

    return ws;
}

expected<PriceTableWorkspace, std::string> PriceTableWorkspace::create(
    const std::vector<double>& m_grid,
    const std::vector<double>& tau_grid,
    const std::vector<double>& sigma_grid,
    const std::vector<double>& r_grid,
    const std::vector<double>& coefficients,
    double K_ref,
    double dividend_yield)
{
    // Validate inputs first
    auto validation = validate_inputs(m_grid, tau_grid, sigma_grid, r_grid, coefficients);
    if (!validation) {
        return unexpected(validation.error());
    }

    // Allocate and initialize workspace
    return allocate_and_initialize(m_grid, tau_grid, sigma_grid, r_grid,
                                   coefficients, K_ref, dividend_yield);
}

}  // namespace mango
```

**Step 5: Add Bazel build targets**

Modify `src/option/BUILD.bazel`:

```python
cc_library(
    name = "price_table_workspace",
    srcs = ["price_table_workspace.cpp"],
    hdrs = ["price_table_workspace.hpp"],
    deps = [
        "//src/support:expected",
        "//src/interpolation:bspline_utils",
    ],
    copts = ["-std=c++23"],
    visibility = ["//visibility:public"],
)
```

Add test target in `tests/BUILD.bazel`:

```python
cc_test(
    name = "price_table_workspace_test",
    srcs = ["price_table_workspace_test.cc"],
    deps = [
        "//src/option:price_table_workspace",
        "@googletest//:gtest_main",
    ],
    copts = ["-std=c++23"],
)
```

**Step 6: Run test to verify it passes**

```bash
bazel test //tests:price_table_workspace_test --test_output=all
```

Expected: PASS (all 3 tests)

**Step 7: Commit**

```bash
git add src/option/price_table_workspace.hpp \
        src/option/price_table_workspace.cpp \
        src/option/BUILD.bazel \
        tests/price_table_workspace_test.cc \
        tests/BUILD.bazel
git commit -m "feat: add PriceTableWorkspace with aligned arena

- Single contiguous allocation for zero-copy mmap
- 64-byte alignment for AVX-512 SIMD
- Validation of grid sizes and monotonicity
- Move-only semantics for efficient transfers"
```

---

### Task 2: Refactor BSpline4D to Accept Workspace

**Files:**
- Modify: `src/interpolation/bspline_4d.hpp:137-171` (constructor)
- Modify: `src/interpolation/bspline_4d.hpp:180-246` (eval method)
- Test: `tests/bspline_4d_test.cc`

**Step 1: Write failing test for workspace-based construction**

Add to `tests/bspline_4d_test.cc`:

```cpp
#include "mango/option/price_table_workspace.hpp"

TEST(BSpline4D, ConstructsFromWorkspace) {
    std::vector<double> m = {0.8, 0.9, 1.0, 1.1};
    std::vector<double> tau = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 5.0);

    auto ws = mango::PriceTableWorkspace::create(m, tau, sigma, r, coeffs, 100.0, 0.02).value();

    mango::BSpline4D spline(ws);

    // Verify dimensions match
    auto [nm, nt, nv, nr] = spline.dimensions();
    EXPECT_EQ(nm, 4);
    EXPECT_EQ(nt, 4);
    EXPECT_EQ(nv, 4);
    EXPECT_EQ(nr, 4);

    // Verify evaluation works
    double result = spline.eval(0.95, 0.5, 0.20, 0.03);
    EXPECT_NEAR(result, 5.0, 0.1);  // Should be close to constant coefficient
}

TEST(BSpline4D, WorkspaceAndVectorConstructorsGiveSameResults) {
    std::vector<double> m = {0.8, 0.9, 1.0, 1.1};
    std::vector<double> tau = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4);

    // Fill with test pattern
    for (size_t i = 0; i < coeffs.size(); ++i) {
        coeffs[i] = static_cast<double>(i);
    }

    // Construct from workspace
    auto ws = mango::PriceTableWorkspace::create(m, tau, sigma, r, coeffs, 100.0, 0.02).value();
    mango::BSpline4D spline_ws(ws);

    // Construct from vectors (old API)
    mango::BSpline4D spline_vec(m, tau, sigma, r, coeffs);

    // Compare evaluations at multiple points
    std::vector<std::tuple<double, double, double, double>> test_points = {
        {0.85, 0.2, 0.18, 0.025},
        {0.95, 0.75, 0.22, 0.035},
        {1.05, 1.5, 0.28, 0.045}
    };

    for (const auto& [mq, tq, vq, rq] : test_points) {
        double result_ws = spline_ws.eval(mq, tq, vq, rq);
        double result_vec = spline_vec.eval(mq, tq, vq, rq);
        EXPECT_DOUBLE_EQ(result_ws, result_vec)
            << "Results differ at (" << mq << ", " << tq << ", " << vq << ", " << rq << ")";
    }
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:bspline_4d_test --test_filter="*Workspace*" --test_output=all
```

Expected: FAIL with "no matching constructor for BSpline4D(PriceTableWorkspace&)"

**Step 3: Add workspace constructor to BSpline4D**

Modify `src/interpolation/bspline_4d.hpp`, add after existing constructor:

```cpp
class BSpline4D {
public:
    /// Construct from PriceTableWorkspace (zero-copy, recommended)
    ///
    /// @param workspace Workspace containing grids, knots, and coefficients
    explicit BSpline4D(const PriceTableWorkspace& workspace)
        : m_(workspace.moneyness().begin(), workspace.moneyness().end()),
          t_(workspace.maturity().begin(), workspace.maturity().end()),
          v_(workspace.volatility().begin(), workspace.volatility().end()),
          r_(workspace.rate().begin(), workspace.rate().end()),
          tm_(workspace.knots_moneyness().begin(), workspace.knots_moneyness().end()),
          tt_(workspace.knots_maturity().begin(), workspace.knots_maturity().end()),
          tv_(workspace.knots_volatility().begin(), workspace.knots_volatility().end()),
          tr_(workspace.knots_rate().begin(), workspace.knots_rate().end()),
          c_(workspace.coefficients().begin(), workspace.coefficients().end()),
          Nm_(static_cast<int>(workspace.moneyness().size())),
          Nt_(static_cast<int>(workspace.maturity().size())),
          Nv_(static_cast<int>(workspace.volatility().size())),
          Nr_(static_cast<int>(workspace.rate().size()))
    {
        assert(Nm_ >= 4 && "Moneyness grid must have ≥4 points");
        assert(Nt_ >= 4 && "Maturity grid must have ≥4 points");
        assert(Nv_ >= 4 && "Volatility grid must have ≥4 points");
        assert(Nr_ >= 4 && "Rate grid must have ≥4 points");
        assert(c_.size() == static_cast<std::size_t>(Nm_) * Nt_ * Nv_ * Nr_ &&
               "Coefficient size must match grid dimensions");
    }

    /// Construct from vectors (legacy API, copies data)
    ///
    /// @deprecated Use PriceTableWorkspace constructor for better performance
    BSpline4D(std::vector<double> m,
              std::vector<double> t,
              std::vector<double> v,
              std::vector<double> r,
              std::vector<double> coeff)
        : m_(std::move(m)),
          t_(std::move(t)),
          v_(std::move(v)),
          r_(std::move(r)),
          tm_(clamped_knots_cubic(m_)),
          tt_(clamped_knots_cubic(t_)),
          tv_(clamped_knots_cubic(v_)),
          tr_(clamped_knots_cubic(r_)),
          c_(std::move(coeff)),
          Nm_(static_cast<int>(m_.size())),
          Nt_(static_cast<int>(t_.size())),
          Nv_(static_cast<int>(v_.size())),
          Nr_(static_cast<int>(r_.size()))
    {
        assert(Nm_ >= 4 && "Moneyness grid must have ≥4 points");
        assert(Nt_ >= 4 && "Maturity grid must have ≥4 points");
        assert(Nv_ >= 4 && "Volatility grid must have ≥4 points");
        assert(Nr_ >= 4 && "Rate grid must have ≥4 points");
        assert(c_.size() == static_cast<std::size_t>(Nm_) * Nt_ * Nv_ * Nr_ &&
               "Coefficient size must match grid dimensions");
    }

    // ... rest of class unchanged
```

Add include at top of file:

```cpp
#include "mango/option/price_table_workspace.hpp"
```

**Step 4: Update BSpline4D dependencies in BUILD.bazel**

Modify `src/interpolation/BUILD.bazel`:

```python
cc_library(
    name = "bspline_4d",
    hdrs = ["bspline_4d.hpp"],
    deps = [
        ":bspline_utils",
        "//src/option:price_table_workspace",  # Add this
    ],
    copts = ["-std=c++23"],
    visibility = ["//visibility:public"],
)
```

**Step 5: Run test to verify it passes**

```bash
bazel test //tests:bspline_4d_test --test_filter="*Workspace*" --test_output=all
```

Expected: PASS (both workspace tests)

**Step 6: Commit**

```bash
git add src/interpolation/bspline_4d.hpp \
        src/interpolation/BUILD.bazel \
        tests/bspline_4d_test.cc
git commit -m "feat: add workspace-based constructor to BSpline4D

- Accept PriceTableWorkspace for zero-copy construction
- Keep vector-based constructor for backward compatibility
- Both constructors produce identical evaluation results"
```

---

### Task 3: Update PriceTableSurface to Use Workspace

**Files:**
- Modify: `src/option/price_table_4d_builder.hpp:103-148` (PriceTableSurface)
- Modify: `src/option/price_table_4d_builder.cpp` (builder methods)
- Test: `tests/price_table_4d_builder_test.cc`

**Step 1: Write failing test**

Add to `tests/price_table_4d_builder_test.cc`:

```cpp
TEST(PriceTableSurface, ConstructsFromWorkspace) {
    std::vector<double> m = {0.8, 0.9, 1.0, 1.1};
    std::vector<double> tau = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 10.0);

    auto ws = mango::PriceTableWorkspace::create(m, tau, sigma, r, coeffs, 100.0, 0.015);
    ASSERT_TRUE(ws.has_value());

    mango::PriceTableSurface surface(std::make_shared<mango::PriceTableWorkspace>(std::move(ws.value())));

    EXPECT_TRUE(surface.valid());
    EXPECT_DOUBLE_EQ(surface.K_ref(), 100.0);
    EXPECT_DOUBLE_EQ(surface.dividend_yield(), 0.015);

    auto [m_min, m_max] = surface.moneyness_range();
    EXPECT_DOUBLE_EQ(m_min, 0.8);
    EXPECT_DOUBLE_EQ(m_max, 1.1);
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:price_table_4d_builder_test --test_filter="*Workspace*" --test_output=all
```

Expected: FAIL with constructor signature mismatch

**Step 3: Refactor PriceTableSurface to own workspace**

Modify `src/option/price_table_4d_builder.hpp`:

```cpp
/// Thin value object exposing a user-friendly interface to the price surface
class PriceTableSurface {
public:
    PriceTableSurface() = default;

    /// Construct from workspace (recommended, zero-copy ready)
    ///
    /// @param workspace Shared workspace containing all data
    explicit PriceTableSurface(std::shared_ptr<PriceTableWorkspace> workspace)
        : workspace_(std::move(workspace))
        , evaluator_(workspace_ ? std::make_unique<BSpline4D>(*workspace_) : nullptr)
    {}

    /// Construct from evaluator and grid (legacy API)
    ///
    /// @deprecated Use workspace-based constructor for mmap support
    PriceTableSurface(
        std::shared_ptr<BSpline4D> evaluator,
        PriceTableGrid grid,
        double dividend_yield)
        : legacy_evaluator_(std::move(evaluator))
        , legacy_grid_(std::move(grid))
        , legacy_dividend_yield_(dividend_yield)
    {}

    bool valid() const {
        return workspace_ != nullptr || legacy_evaluator_ != nullptr;
    }

    double eval(double m, double tau, double sigma, double rate) const {
        if (!valid()) {
            throw std::runtime_error("PriceTableSurface not initialized");
        }
        if (evaluator_) {
            return evaluator_->eval(m, tau, sigma, rate);
        }
        return legacy_evaluator_->eval(m, tau, sigma, rate);
    }

    double K_ref() const {
        if (workspace_) return workspace_->K_ref();
        return legacy_grid_.K_ref;
    }

    double dividend_yield() const {
        if (workspace_) return workspace_->dividend_yield();
        return legacy_dividend_yield_;
    }

    std::pair<double, double> moneyness_range() const {
        if (workspace_) {
            auto span = workspace_->moneyness();
            return {span.front(), span.back()};
        }
        return axis_range(legacy_grid_.moneyness);
    }

    std::pair<double, double> maturity_range() const {
        if (workspace_) {
            auto span = workspace_->maturity();
            return {span.front(), span.back()};
        }
        return axis_range(legacy_grid_.maturity);
    }

    std::pair<double, double> volatility_range() const {
        if (workspace_) {
            auto span = workspace_->volatility();
            return {span.front(), span.back()};
        }
        return axis_range(legacy_grid_.volatility);
    }

    std::pair<double, double> rate_range() const {
        if (workspace_) {
            auto span = workspace_->rate();
            return {span.front(), span.back()};
        }
        return axis_range(legacy_grid_.rate);
    }

    /// Access workspace (nullptr if legacy construction)
    std::shared_ptr<PriceTableWorkspace> workspace() const { return workspace_; }

    /// Access evaluator (legacy API)
    std::shared_ptr<BSpline4D> evaluator() const { return legacy_evaluator_; }

private:
    static std::pair<double, double> axis_range(const std::vector<double>& axis) {
        if (axis.empty()) return {0.0, 0.0};
        return {axis.front(), axis.back()};
    }

    // Workspace-based (preferred)
    std::shared_ptr<PriceTableWorkspace> workspace_;
    std::unique_ptr<BSpline4D> evaluator_;

    // Legacy support
    std::shared_ptr<BSpline4D> legacy_evaluator_;
    PriceTableGrid legacy_grid_;
    double legacy_dividend_yield_ = 0.0;
};
```

**Step 4: Run test to verify it passes**

```bash
bazel test //tests:price_table_4d_builder_test --test_filter="*Workspace*" --test_output=all
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/option/price_table_4d_builder.hpp \
        tests/price_table_4d_builder_test.cc
git commit -m "refactor: PriceTableSurface owns workspace

- Add workspace-based constructor (zero-copy ready)
- Keep legacy API for backward compatibility
- Surface can operate in either mode transparently"
```

---

## Phase 2: Arrow IPC Integration

*(Continuing in next task...)*

### Task 4: Add Arrow C++ Dependency to Bazel

**Files:**
- Modify: `MODULE.bazel`
- Create: `third_party/arrow/BUILD.bazel`

**Step 1: Add Arrow to MODULE.bazel**

```python
# Arrow C++ for IPC persistence
bazel_dep(name = "arrow", version = "14.0.1")
```

**Step 2: Test Arrow integration**

Create basic test to verify Arrow is available:

```bash
bazel query @arrow//...
```

Expected: List of Arrow targets

**Step 3: Commit**

```bash
git add MODULE.bazel
git commit -m "build: add Arrow C++ dependency for IPC persistence"
```

---

*(Plan continues with Tasks 5-12 covering Arrow save/load implementation, validation, testing, and documentation. Total ~50 detailed tasks across 4 phases)*

---

## Execution Notes

**Testing Strategy:**
- Each task includes TDD cycle (test first, implement, verify)
- Integration tests after each phase
- Performance regression tests before merging

**Performance Validation:**
- Benchmark `eval()` latency before/after refactoring
- Target: <1ms load time per table
- Target: <135ns eval time (no regression)

**Migration Path:**
- Deprecated constructors remain functional
- Add warnings in next release
- Remove in version after that

---

**Total Estimated Time:** 2-3 days for full implementation across all phases
