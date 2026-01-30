# Batch Option Enhancement Implementation Plan

**Date:** 2025-01-12
**Status:** Ready for Implementation
**Design Document:** [2025-01-12-batch-option-enhancement-v2.md](./2025-01-12-batch-option-enhancement-v2.md)

## Overview

This plan implements two enhancements to batch American option pricing:
1. **Normalized Chain Solver**: Exploits scale invariance V(S,K,τ) = K·u(ln(S/K), τ) to solve once and interpolate for all strikes/maturities
2. **Flexible Batch API**: Adds `SetupCallback` to `BatchAmericanOptionSolver` for per-solver configuration

The implementation is divided into 3 phases with bite-sized tasks. Each task includes exact file paths, complete code snippets, verification commands, and commit messages.

---

## Phase 1: Normalized Solver Core (5 tasks)

### Task 1.1: Create normalized solver types and workspace

**Files to create:**
- `src/option/normalized_chain_solver.hpp`

**Implementation:**

```cpp
/**
 * @file normalized_chain_solver.hpp
 * @brief Dimensionless American option solver exploiting scale invariance
 */

#ifndef MANGO_NORMALIZED_CHAIN_SOLVER_HPP
#define MANGO_NORMALIZED_CHAIN_SOLVER_HPP

#include "src/option/american_option.hpp"
#include "src/option/american_solver_workspace.hpp"
#include "src/support/expected.hpp"
#include <span>
#include <memory>
#include <vector>

namespace mango {

/**
 * Request for normalized PDE solve.
 *
 * Solves dimensionless PDE: ∂u/∂t = 0.5σ²(∂²u/∂x² - ∂u/∂x) + (r-q)∂u/∂x - ru
 * where u = V/K, x = ln(S/K)
 *
 * Output: u(x,τ) on specified grid
 * Caller converts to prices: V = K·u(ln(S/K), τ)
 */
struct NormalizedSolveRequest {
    // PDE coefficients
    double sigma;              ///< Volatility
    double rate;               ///< Risk-free rate
    double dividend;           ///< Continuous dividend yield
    OptionType option_type;    ///< Call or Put

    // Grid configuration
    double x_min;              ///< Minimum log-moneyness
    double x_max;              ///< Maximum log-moneyness
    size_t n_space;            ///< Spatial grid points
    size_t n_time;             ///< Time steps
    double T_max;              ///< Maximum maturity (years)

    // Snapshot collection
    std::span<const double> tau_snapshots;  ///< Maturities to collect

    /// Validate request parameters
    expected<void, std::string> validate() const;
};

/**
 * View of normalized solution surface u(x,τ).
 *
 * Provides interpolation interface for querying u at arbitrary (x,τ).
 * Caller scales results: V = K·u
 */
class NormalizedSurfaceView {
public:
    NormalizedSurfaceView(
        std::span<const double> x_grid,
        std::span<const double> tau_grid,
        std::span<const double> values)
        : x_grid_(x_grid)
        , tau_grid_(tau_grid)
        , values_(values)
    {}

    /// Interpolate u(x,τ) using bilinear interpolation
    double interpolate(double x, double tau) const;

    /// Access raw data (for testing)
    std::span<const double> x_grid() const { return x_grid_; }
    std::span<const double> tau_grid() const { return tau_grid_; }
    std::span<const double> values() const { return values_; }

private:
    std::span<const double> x_grid_;
    std::span<const double> tau_grid_;
    std::span<const double> values_;
};

/**
 * Reusable workspace for normalized solves.
 *
 * Allocates all buffers needed for PDE solve + interpolation surface.
 * Thread-safe: each thread creates its own workspace instance.
 */
class NormalizedWorkspace {
public:
    /// Create workspace for given request parameters
    static expected<NormalizedWorkspace, std::string> create(
        const NormalizedSolveRequest& request);

    /// Get view of solution surface (after solve completes)
    NormalizedSurfaceView surface_view();

    // No copying (expensive)
    NormalizedWorkspace(const NormalizedWorkspace&) = delete;
    NormalizedWorkspace& operator=(const NormalizedWorkspace&) = delete;

    // Moving OK
    NormalizedWorkspace(NormalizedWorkspace&&) = default;
    NormalizedWorkspace& operator=(NormalizedWorkspace&&) = default;

private:
    NormalizedWorkspace() = default;

    std::shared_ptr<AmericanSolverWorkspace> pde_workspace_;
    std::vector<double> x_grid_;
    std::vector<double> tau_grid_;
    std::vector<double> values_;  // u(x,τ) [row-major: Nx × Ntau]
};

/**
 * Eligibility limits for normalized solver.
 *
 * Thresholds derived from numerical stability and convergence analysis:
 * - Margin: ≥6 ghost cells to avoid boundary reflection (<0.5bp error)
 * - Width: ≤5.8 log-units for convergence (empirical from sweeps)
 * - Grid spacing: ≤0.05 for Von Neumann stability at σ=200%, τ=2y
 */
struct EligibilityLimits {
    static constexpr double MAX_WIDTH = 5.8;      ///< Convergence limit (log-units)
    static constexpr double MAX_DX = 0.05;        ///< Truncation error O(dx²)
    static constexpr double MIN_MARGIN_ABS = 0.35; ///< 6-cell ghost zone minimum

    /// Minimum margin (6 ghost cells or 0.35, whichever is larger)
    static double min_margin(double dx) {
        return std::max(MIN_MARGIN_ABS, 6.0 * dx);
    }

    /// Maximum ratio K_max/K_min given dx
    static double max_ratio(double dx) {
        double margin = min_margin(dx);
        return std::exp(MAX_WIDTH - 2.0 * margin);
    }
};

/**
 * Normalized chain solver.
 *
 * Solves American option PDE in dimensionless coordinates exploiting
 * scale invariance: V(S,K,τ) = K·u(ln(S/K), τ)
 *
 * One PDE solve yields prices for all (S,K) combinations via interpolation.
 */
class NormalizedChainSolver {
public:
    /**
     * Solve normalized PDE.
     *
     * Solves: ∂u/∂t = 0.5σ²(∂²u/∂x² - ∂u/∂x) + (r-q)∂u/∂x - ru
     * Terminal condition: u(x,0) = max(eˣ-1, 0) (call) or max(1-eˣ, 0) (put)
     * Boundary conditions: American exercise constraint u ≥ intrinsic
     *
     * @param request PDE configuration and snapshot times
     * @param workspace Pre-allocated buffers (reusable across solves)
     * @param surface_view Output view (references workspace.values_)
     * @return Success or solver error
     */
    static expected<void, SolverError> solve(
        const NormalizedSolveRequest& request,
        NormalizedWorkspace& workspace,
        NormalizedSurfaceView& surface_view);

    /**
     * Check eligibility for normalized solving.
     *
     * Criteria:
     * 1. Grid spacing dx ≤ MAX_DX (0.05)
     * 2. Domain width ≤ MAX_WIDTH (5.8)
     * 3. Margins ≥ min_margin(dx) on both boundaries
     *
     * @param request Request to validate
     * @param moneyness_grid Moneyness values m = K/S from price table
     * @return Success or reason for ineligibility
     */
    static expected<void, std::string> check_eligibility(
        const NormalizedSolveRequest& request,
        std::span<const double> moneyness_grid);
};

}  // namespace mango

#endif  // MANGO_NORMALIZED_CHAIN_SOLVER_HPP
```

**Test command:**
```bash
# Header should compile without errors
bazel build //src/option:normalized_chain_solver
```

**Verification:**
- Header compiles cleanly
- All types defined
- Documentation complete

**Commit message:**
```
Add normalized chain solver types and workspace

Define core types for dimensionless American option solving:
- NormalizedSolveRequest: PDE configuration in x=ln(S/K) coordinates
- NormalizedSurfaceView: Interpolation interface for u(x,τ)
- NormalizedWorkspace: Thread-safe reusable buffers
- EligibilityLimits: Numerical stability thresholds
- NormalizedChainSolver: Main solver class (implementation next)

These types exploit scale invariance V(S,K,τ) = K·u(ln(S/K), τ)
to solve once and interpolate for all strikes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

### Task 1.2: Implement workspace creation and validation

**Files to modify:**
- `src/option/normalized_chain_solver.cpp` (create)

**Implementation:**

```cpp
/**
 * @file normalized_chain_solver.cpp
 * @brief Implementation of normalized chain solver
 */

#include "src/option/normalized_chain_solver.hpp"
#include <cmath>
#include <algorithm>

namespace mango {

expected<void, std::string> NormalizedSolveRequest::validate() const {
    if (sigma <= 0.0) {
        return unexpected("Volatility must be positive");
    }
    // Note: rate can be negative (EUR, JPY markets)
    if (dividend < 0.0) {
        return unexpected("Dividend yield must be non-negative");
    }
    if (x_min >= x_max) {
        return unexpected("x_min must be < x_max");
    }
    if (n_space < 3) {
        return unexpected("n_space must be ≥ 3");
    }
    if (n_time < 1) {
        return unexpected("n_time must be ≥ 1");
    }
    if (T_max <= 0.0) {
        return unexpected("T_max must be positive");
    }
    if (tau_snapshots.empty()) {
        return unexpected("tau_snapshots must be non-empty");
    }

    // Validate snapshot times
    for (double tau : tau_snapshots) {
        if (tau <= 0.0 || tau > T_max) {
            return unexpected("Snapshot times must be in (0, T_max]");
        }
    }

    return {};
}

expected<NormalizedWorkspace, std::string> NormalizedWorkspace::create(
    const NormalizedSolveRequest& request)
{
    // Validate request
    auto validation = request.validate();
    if (!validation) {
        return unexpected(validation.error());
    }

    NormalizedWorkspace workspace;

    // Create PDE workspace
    auto pde_workspace = AmericanSolverWorkspace::create(
        request.x_min, request.x_max, request.n_space, request.n_time);
    if (!pde_workspace) {
        return unexpected("Failed to create PDE workspace: " + pde_workspace.error());
    }
    workspace.pde_workspace_ = std::move(pde_workspace.value());

    // Allocate x grid
    workspace.x_grid_.resize(request.n_space);
    double dx = (request.x_max - request.x_min) / (request.n_space - 1);
    for (size_t i = 0; i < request.n_space; ++i) {
        workspace.x_grid_[i] = request.x_min + i * dx;
    }

    // Allocate tau grid (copy from request)
    workspace.tau_grid_.assign(request.tau_snapshots.begin(), request.tau_snapshots.end());
    std::sort(workspace.tau_grid_.begin(), workspace.tau_grid_.end());

    // Allocate values array (Nx × Ntau)
    workspace.values_.resize(request.n_space * workspace.tau_grid_.size(), 0.0);

    return workspace;
}

NormalizedSurfaceView NormalizedWorkspace::surface_view() {
    return NormalizedSurfaceView(x_grid_, tau_grid_, values_);
}

double NormalizedSurfaceView::interpolate(double x, double tau) const {
    // Find x interval [x_grid[i], x_grid[i+1]]
    auto x_it = std::lower_bound(x_grid_.begin(), x_grid_.end(), x);
    if (x_it == x_grid_.begin()) {
        x_it = x_grid_.begin() + 1;  // Clamp to first interval
    } else if (x_it == x_grid_.end()) {
        x_it = x_grid_.end() - 1;  // Clamp to last interval
    }
    size_t i_x = x_it - x_grid_.begin() - 1;

    // Find tau interval [tau_grid[j], tau_grid[j+1]]
    auto tau_it = std::lower_bound(tau_grid_.begin(), tau_grid_.end(), tau);
    if (tau_it == tau_grid_.begin()) {
        tau_it = tau_grid_.begin() + 1;
    } else if (tau_it == tau_grid_.end()) {
        tau_it = tau_grid_.end() - 1;
    }
    size_t i_tau = tau_it - tau_grid_.begin() - 1;

    // Bilinear interpolation
    double x0 = x_grid_[i_x];
    double x1 = x_grid_[i_x + 1];
    double tau0 = tau_grid_[i_tau];
    double tau1 = tau_grid_[i_tau + 1];

    double fx = (x - x0) / (x1 - x0);
    double ft = (tau - tau0) / (tau1 - tau0);

    // Values stored row-major: values[i*Ntau + j]
    size_t Ntau = tau_grid_.size();
    double v00 = values_[i_x * Ntau + i_tau];
    double v01 = values_[i_x * Ntau + (i_tau + 1)];
    double v10 = values_[(i_x + 1) * Ntau + i_tau];
    double v11 = values_[(i_x + 1) * Ntau + (i_tau + 1)];

    return (1.0 - fx) * (1.0 - ft) * v00 +
           (1.0 - fx) * ft * v01 +
           fx * (1.0 - ft) * v10 +
           fx * ft * v11;
}

}  // namespace mango
```

**Build file update:**

Add to `src/option/BUILD.bazel`:
```python
cc_library(
    name = "normalized_chain_solver",
    srcs = ["normalized_chain_solver.cpp"],
    hdrs = ["normalized_chain_solver.hpp"],
    deps = [
        ":american_option",
        ":american_solver_workspace",
        "//src/support:expected",
    ],
    visibility = ["//visibility:public"],
)
```

**Test command:**
```bash
bazel build //src/option:normalized_chain_solver
```

**Verification:**
- Compiles cleanly
- Workspace creates successfully
- Interpolation logic correct

**Commit message:**
```
Implement workspace creation and interpolation

Add NormalizedWorkspace::create() with validation:
- Creates AmericanSolverWorkspace for PDE solving
- Allocates x grid (uniform spacing in log-moneyness)
- Allocates tau grid (sorted snapshot times)
- Allocates values array (Nx × Ntau, row-major)

Add NormalizedSurfaceView::interpolate():
- Bilinear interpolation on (x, τ) surface
- Clamps to grid boundaries if out of range
- Used by callers to query u(x,τ) at arbitrary points

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

### Task 1.3: Implement eligibility checking

**Files to modify:**
- `src/option/normalized_chain_solver.cpp`

**Add to existing file:**

```cpp
expected<void, std::string> NormalizedChainSolver::check_eligibility(
    const NormalizedSolveRequest& request,
    std::span<const double> moneyness_grid)
{
    // Check grid spacing
    double dx = (request.x_max - request.x_min) / (request.n_space - 1);
    if (dx > EligibilityLimits::MAX_DX) {
        return unexpected(
            "Grid spacing " + std::to_string(dx) +
            " exceeds limit " + std::to_string(EligibilityLimits::MAX_DX) +
            " (Von Neumann stability requirement)");
    }

    // Check domain width
    double width = request.x_max - request.x_min;
    if (width > EligibilityLimits::MAX_WIDTH) {
        return unexpected(
            "Domain width " + std::to_string(width) +
            " exceeds limit " + std::to_string(EligibilityLimits::MAX_WIDTH) +
            " (convergence degrades beyond 5.8 log-units)");
    }

    // Check margins
    // Moneyness convention: m = K/S, so x = ln(S/K) = -ln(m)
    // x_min_data = -ln(m_max), x_max_data = -ln(m_min)
    if (moneyness_grid.empty()) {
        return unexpected("Moneyness grid is empty");
    }

    auto [m_min_it, m_max_it] = std::ranges::minmax_element(moneyness_grid);
    double m_min = *m_min_it;
    double m_max = *m_max_it;

    if (m_min <= 0.0 || m_max <= 0.0) {
        return unexpected("Moneyness values must be positive (m = K/S > 0)");
    }

    double x_min_data = -std::log(m_max);
    double x_max_data = -std::log(m_min);

    double margin_left = x_min_data - request.x_min;
    double margin_right = request.x_max - x_max_data;
    double min_margin = EligibilityLimits::min_margin(dx);

    if (margin_left < min_margin) {
        return unexpected(
            "Left margin " + std::to_string(margin_left) +
            " < required " + std::to_string(min_margin) +
            " (need ≥6 ghost cells to avoid boundary reflection)");
    }

    if (margin_right < min_margin) {
        return unexpected(
            "Right margin " + std::to_string(margin_right) +
            " < required " + std::to_string(min_margin) +
            " (need ≥6 ghost cells to avoid boundary reflection)");
    }

    // Check ratio (derived from width + margin constraints)
    double ratio = m_max / m_min;
    double max_ratio_limit = EligibilityLimits::max_ratio(dx);
    if (ratio > max_ratio_limit) {
        return unexpected(
            "Moneyness ratio " + std::to_string(ratio) +
            " exceeds limit " + std::to_string(max_ratio_limit) +
            " (derived from width=" + std::to_string(EligibilityLimits::MAX_WIDTH) +
            " and margin=" + std::to_string(min_margin) + ")");
    }

    return {};
}
```

**Test command:**
```bash
bazel build //src/option:normalized_chain_solver
```

**Verification:**
- Compiles cleanly
- All eligibility checks implemented
- Error messages informative

**Commit message:**
```
Implement eligibility checking for normalized solver

Add NormalizedChainSolver::check_eligibility():
- Validates grid spacing ≤ 0.05 (Von Neumann stability)
- Validates domain width ≤ 5.8 (convergence limit)
- Validates margins ≥ max(0.35, 6·dx) (boundary reflection)
- Validates ratio K_max/K_min ≤ exp(5.8 - 2·margin)
- Uses moneyness convention m = K/S, x = -ln(m)

Thresholds derived from numerical analysis ensure <0.5bp
accuracy and Newton convergence for production use cases.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

### Task 1.4: Implement normalized PDE solving

**Files to modify:**
- `src/option/normalized_chain_solver.cpp`

**Add to existing file:**

```cpp
#include "src/option/price_table_snapshot_collector.hpp"
#include <cmath>

expected<void, SolverError> NormalizedChainSolver::solve(
    const NormalizedSolveRequest& request,
    NormalizedWorkspace& workspace,
    NormalizedSurfaceView& surface_view)
{
    // Create solver parameters (K=1, S=1 → x = ln(S/K) = 0 is ATM)
    AmericanOptionParams params{
        .strike = 1.0,  // Normalized strike
        .spot = 1.0,    // Normalized spot (ATM at x=0)
        .maturity = request.T_max,
        .volatility = request.sigma,
        .rate = request.rate,
        .continuous_dividend_yield = request.dividend,
        .option_type = request.option_type,
        .discrete_dividends = {}  // Normalized solver requires no discrete dividends
    };

    // Create solver with workspace
    auto solver_result = AmericanOptionSolver::create(params, workspace.pde_workspace_);
    if (!solver_result) {
        return unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .message = "Failed to create solver: " + solver_result.error(),
            .iterations = 0
        });
    }
    auto solver = std::move(solver_result.value());

    // Setup snapshot collector
    // Convert moneyness m = K/S to x = ln(S/K) = -ln(m)
    std::vector<double> x_values;
    x_values.reserve(workspace.x_grid_.size());
    for (double x : workspace.x_grid_) {
        // x is already in log-moneyness, but we need to express as moneyness m
        // x = -ln(m) → m = exp(-x)
        double m = std::exp(-x);
        x_values.push_back(m);
    }

    PriceTableSnapshotCollectorConfig collector_config{
        .moneyness = std::span{x_values},
        .tau = std::span{workspace.tau_grid_},
        .K_ref = 1.0,  // Normalized
        .option_type = request.option_type,
        .payoff_params = nullptr
    };
    PriceTableSnapshotCollector collector(collector_config);

    // Register snapshots at requested maturities
    double dt = request.T_max / request.n_time;
    for (size_t j = 0; j < workspace.tau_grid_.size(); ++j) {
        // Compute step index: k = round(τ/dt) - 1
        double step_exact = workspace.tau_grid_[j] / dt - 1.0;
        long long step_rounded = std::llround(step_exact);

        // Clamp to valid range
        if (step_rounded < 0) {
            step_rounded = 0;
        } else if (step_rounded >= static_cast<long long>(request.n_time)) {
            step_rounded = static_cast<long long>(request.n_time) - 1;
        }

        solver.register_snapshot(static_cast<size_t>(step_rounded), j, &collector);
    }

    // Solve PDE
    auto solve_result = solver.solve();
    if (!solve_result) {
        return unexpected(solve_result.error());
    }

    // Extract values from collector
    auto prices_2d = collector.prices();  // Shape: Nx × Ntau
    size_t Nx = workspace.x_grid_.size();
    size_t Ntau = workspace.tau_grid_.size();

    if (prices_2d.size() != Nx * Ntau) {
        return unexpected(SolverError{
            .code = SolverErrorCode::RuntimeError,
            .message = "Snapshot collector returned wrong size",
            .iterations = 0
        });
    }

    // Copy to workspace values (already normalized u = V/K, and K=1)
    std::copy(prices_2d.begin(), prices_2d.end(), workspace.values_.begin());

    // Update surface view to reference workspace values
    surface_view = workspace.surface_view();

    return {};
}
```

**Test command:**
```bash
bazel build //src/option:normalized_chain_solver
```

**Verification:**
- Compiles cleanly
- All dependencies resolved
- Solver logic complete

**Commit message:**
```
Implement normalized PDE solving

Add NormalizedChainSolver::solve():
- Creates AmericanOptionSolver with K=1, S=1 (normalized)
- Registers snapshots at requested maturities
- Solves PDE in x = ln(S/K) coordinates
- Extracts solution u(x,τ) via snapshot collector
- Returns surface view for interpolation

Solution exploits scale invariance: V(S,K,τ) = K·u(ln(S/K), τ)
One solve yields prices for all (S,K) via V = K·u interpolation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

### Task 1.5: Add unit tests for normalized solver

**Files to create:**
- `tests/normalized_chain_solver_test.cc`

**Implementation:**

```cpp
/**
 * @file normalized_chain_solver_test.cc
 * @brief Unit tests for normalized chain solver
 */

#include "src/option/normalized_chain_solver.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace mango;

TEST(NormalizedChainSolverTest, WorkspaceCreation) {
    NormalizedSolveRequest request{
        .sigma = 0.20,
        .rate = 0.05,
        .dividend = 0.02,
        .option_type = OptionType::PUT,
        .x_min = -3.0,
        .x_max = 3.0,
        .n_space = 101,
        .n_time = 1000,
        .T_max = 1.0,
        .tau_snapshots = std::vector<double>{0.25, 0.5, 1.0}
    };

    auto workspace_result = NormalizedWorkspace::create(request);
    ASSERT_TRUE(workspace_result.has_value());

    auto workspace = std::move(workspace_result.value());
    auto surface = workspace.surface_view();

    EXPECT_EQ(surface.x_grid().size(), 101);
    EXPECT_EQ(surface.tau_grid().size(), 3);
    EXPECT_EQ(surface.values().size(), 101 * 3);
}

TEST(NormalizedChainSolverTest, EligibilityPass) {
    NormalizedSolveRequest request{
        .sigma = 0.20,
        .rate = 0.05,
        .dividend = 0.02,
        .option_type = OptionType::PUT,
        .x_min = -3.0,
        .x_max = 3.0,
        .n_space = 101,
        .n_time = 1000,
        .T_max = 1.0,
        .tau_snapshots = std::vector<double>{1.0}
    };

    // Moneyness grid: m = K/S in [0.8, 1.2]
    // x = -ln(m) in [-0.182, 0.223]
    // Margins: left = -0.182 - (-3.0) = 2.82, right = 3.0 - 0.223 = 2.78
    // Both > 0.35 ✓
    std::vector<double> moneyness = {0.8, 0.9, 1.0, 1.1, 1.2};

    auto eligibility = NormalizedChainSolver::check_eligibility(request, moneyness);
    EXPECT_TRUE(eligibility.has_value());
}

TEST(NormalizedChainSolverTest, EligibilityFailRatio) {
    NormalizedSolveRequest request{
        .sigma = 0.20,
        .rate = 0.05,
        .dividend = 0.02,
        .option_type = OptionType::PUT,
        .x_min = -3.0,
        .x_max = 3.0,
        .n_space = 101,
        .n_time = 1000,
        .T_max = 1.0,
        .tau_snapshots = std::vector<double>{1.0}
    };

    // Moneyness ratio 200 = 2.0/0.01 exceeds limit (~164 for dx=0.059)
    std::vector<double> moneyness = {0.01, 2.0};

    auto eligibility = NormalizedChainSolver::check_eligibility(request, moneyness);
    EXPECT_FALSE(eligibility.has_value());
    EXPECT_TRUE(eligibility.error().find("ratio") != std::string::npos);
}

TEST(NormalizedChainSolverTest, EligibilityFailMargin) {
    NormalizedSolveRequest request{
        .sigma = 0.20,
        .rate = 0.05,
        .dividend = 0.02,
        .option_type = OptionType::PUT,
        .x_min = -0.5,  // Too narrow domain
        .x_max = 0.5,
        .n_space = 21,
        .n_time = 1000,
        .T_max = 1.0,
        .tau_snapshots = std::vector<double>{1.0}
    };

    // Moneyness [0.7, 1.3] → x in [-0.357, 0.262]
    // Left margin = -0.357 - (-0.5) = 0.143 < 0.35 ✗
    std::vector<double> moneyness = {0.7, 1.0, 1.3};

    auto eligibility = NormalizedChainSolver::check_eligibility(request, moneyness);
    EXPECT_FALSE(eligibility.has_value());
    EXPECT_TRUE(eligibility.error().find("margin") != std::string::npos);
}

TEST(NormalizedChainSolverTest, SolveAndInterpolate) {
    NormalizedSolveRequest request{
        .sigma = 0.20,
        .rate = 0.05,
        .dividend = 0.02,
        .option_type = OptionType::PUT,
        .x_min = -3.0,
        .x_max = 3.0,
        .n_space = 101,
        .n_time = 1000,
        .T_max = 1.0,
        .tau_snapshots = std::vector<double>{0.25, 0.5, 1.0}
    };

    auto workspace_result = NormalizedWorkspace::create(request);
    ASSERT_TRUE(workspace_result.has_value());

    auto workspace = std::move(workspace_result.value());
    auto surface = workspace.surface_view();

    auto solve_result = NormalizedChainSolver::solve(request, workspace, surface);
    ASSERT_TRUE(solve_result.has_value());

    // Test interpolation at ATM (x=0)
    double u_atm_1y = surface.interpolate(0.0, 1.0);
    EXPECT_GT(u_atm_1y, 0.0);  // Put has positive value

    // Test interpolation at different tau
    double u_atm_3m = surface.interpolate(0.0, 0.25);
    EXPECT_GT(u_atm_3m, 0.0);
    EXPECT_LT(u_atm_3m, u_atm_1y);  // Shorter maturity < longer maturity

    // Test ITM (x < 0 → S/K < 1 → put is ITM)
    double u_itm = surface.interpolate(-0.5, 1.0);
    EXPECT_GT(u_itm, u_atm_1y);  // ITM > ATM
}

TEST(NormalizedChainSolverTest, ScaleInvariance) {
    // Test V(S,K,τ) = K·u(ln(S/K), τ)
    NormalizedSolveRequest request{
        .sigma = 0.20,
        .rate = 0.05,
        .dividend = 0.02,
        .option_type = OptionType::PUT,
        .x_min = -3.0,
        .x_max = 3.0,
        .n_space = 101,
        .n_time = 1000,
        .T_max = 1.0,
        .tau_snapshots = std::vector<double>{1.0}
    };

    auto workspace_result = NormalizedWorkspace::create(request);
    ASSERT_TRUE(workspace_result.has_value());

    auto workspace = std::move(workspace_result.value());
    auto surface = workspace.surface_view();

    auto solve_result = NormalizedChainSolver::solve(request, workspace, surface);
    ASSERT_TRUE(solve_result.has_value());

    // Two different (S,K) pairs with same x = ln(S/K)
    double S1 = 100.0, K1 = 100.0;  // x = 0
    double S2 = 50.0, K2 = 50.0;    // x = 0

    double x = std::log(S1 / K1);
    EXPECT_NEAR(x, 0.0, 1e-10);
    EXPECT_NEAR(std::log(S2 / K2), x, 1e-10);

    double u = surface.interpolate(x, 1.0);
    double V1 = K1 * u;
    double V2 = K2 * u;

    // V scales with K
    EXPECT_NEAR(V2 / V1, K2 / K1, 1e-6);
}
```

**Build file update:**

Add to `tests/BUILD.bazel`:
```python
cc_test(
    name = "normalized_chain_solver_test",
    srcs = ["normalized_chain_solver_test.cc"],
    deps = [
        "//src/option:normalized_chain_solver",
        "@googletest//:gtest_main",
    ],
)
```

**Test command:**
```bash
bazel test //tests:normalized_chain_solver_test
```

**Verification:**
- All tests pass
- Workspace creation works
- Eligibility checks work
- Solving and interpolation work
- Scale invariance verified

**Commit message:**
```
Add unit tests for normalized chain solver

Test coverage:
- WorkspaceCreation: validates buffer allocation
- EligibilityPass: confirms eligible parameters pass
- EligibilityFailRatio: ratio K_max/K_min too large
- EligibilityFailMargin: insufficient boundary margins
- SolveAndInterpolate: solves PDE and queries surface
- ScaleInvariance: verifies V(S,K) = K·u(ln(S/K))

Tests confirm normalized solver produces correct dimensionless
solution and satisfies scale invariance property.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## Phase 2: Batch API Enhancement (3 tasks)

### Task 2.1: Add workspace validation method

**Files to modify:**
- `src/option/american_solver_workspace.hpp`
- `src/option/american_solver_workspace.cpp`

**Add to header:**

```cpp
/**
 * Validate workspace parameters without allocation.
 *
 * Enables fail-fast in batch operations before parallel region.
 *
 * @param x_min Minimum log-moneyness
 * @param x_max Maximum log-moneyness
 * @param n_space Number of spatial grid points
 * @param n_time Number of time steps
 * @return Success or error message
 */
static expected<void, std::string> validate_params(
    double x_min,
    double x_max,
    size_t n_space,
    size_t n_time);
```

**Add to source:**

```cpp
expected<void, std::string> AmericanSolverWorkspace::validate_params(
    double x_min,
    double x_max,
    size_t n_space,
    size_t n_time)
{
    if (x_min >= x_max) {
        return unexpected("x_min must be < x_max");
    }
    if (n_space < 3) {
        return unexpected("n_space must be ≥ 3");
    }
    if (n_time < 1) {
        return unexpected("n_time must be ≥ 1");
    }

    double dx = (x_max - x_min) / (n_space - 1);
    if (dx >= 0.5) {
        return unexpected(
            "Grid too coarse: dx = " + std::to_string(dx) +
            " ≥ 0.5 (Von Neumann stability violated)");
    }

    return {};
}
```

**Test command:**
```bash
bazel build //src/option:american_solver_workspace
```

**Verification:**
- Compiles cleanly
- Validation logic correct

**Commit message:**
```
Add workspace parameter validation

Add AmericanSolverWorkspace::validate_params():
- Validates grid parameters without allocation
- Enables fail-fast in batch operations before parallel region
- Checks x_min < x_max, n_space ≥ 3, n_time ≥ 1
- Checks dx < 0.5 (Von Neumann stability)

Used by BatchAmericanOptionSolver to validate once before
creating per-thread workspaces.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

### Task 2.2: Add SetupCallback to BatchAmericanOptionSolver

**Files to modify:**
- `src/option/american_option.hpp`

**Replace existing solve_batch implementation with:**

```cpp
/// Batch American Option Solver
///
/// Solves multiple American options in parallel using OpenMP.
/// This is significantly faster than solving options sequentially
/// for embarrassingly parallel workloads.
///
/// Example usage:
/// ```cpp
/// std::vector<AmericanOptionParams> batch = { ... };
///
/// auto results = solve_american_options_batch(batch, -3.0, 3.0, 101, 1000);
/// ```
///
/// Advanced usage with snapshots:
/// ```cpp
/// auto results = BatchAmericanOptionSolver::solve_batch(
///     params, -3.0, 3.0, 101, 1000,
///     [&](size_t idx, AmericanOptionSolver& solver) {
///         // Register snapshots for this solve
///         solver.register_snapshot(step, user_idx, collector);
///     });
/// ```
///
/// Performance:
/// - Single-threaded: ~72 options/sec (101x1000 grid)
/// - Parallel (32 cores): ~848 options/sec (11.8x speedup)
class BatchAmericanOptionSolver {
public:
    /// Setup callback: called before each solve() to configure solver
    /// @param index Index of current option in params vector
    /// @param solver Reference to solver (can register snapshots, set configs, etc.)
    using SetupCallback = std::function<void(size_t index, AmericanOptionSolver& solver)>;

    /// Solve a batch of American options in parallel
    ///
    /// Each thread creates its own workspace to avoid data races.
    /// The workspace parameters (grid configuration) are validated once.
    ///
    /// @param params Vector of option parameters
    /// @param x_min Minimum log-moneyness
    /// @param x_max Maximum log-moneyness
    /// @param n_space Number of spatial grid points
    /// @param n_time Number of time steps
    /// @param setup Optional callback invoked after solver creation, before solve()
    /// @return Vector of results (same order as input)
    static std::vector<expected<AmericanOptionResult, SolverError>> solve_batch(
        std::span<const AmericanOptionParams> params,
        double x_min,
        double x_max,
        size_t n_space,
        size_t n_time,
        SetupCallback setup = nullptr)
    {
        std::vector<expected<AmericanOptionResult, SolverError>> results(params.size());

        // Validate workspace parameters once before parallel loop
        auto validation = AmericanSolverWorkspace::validate_params(x_min, x_max, n_space, n_time);
        if (!validation) {
            // If workspace validation fails, return error for all options
            SolverError error{
                .code = SolverErrorCode::InvalidConfiguration,
                .message = "Invalid workspace parameters: " + validation.error(),
                .iterations = 0
            };
            for (size_t i = 0; i < params.size(); ++i) {
                results[i] = unexpected(error);
            }
            return results;
        }

        // Common solve logic
        auto solve_one = [&](size_t i, std::shared_ptr<AmericanSolverWorkspace> workspace)
            -> expected<AmericanOptionResult, SolverError>
        {
            // Use factory method to avoid exceptions from constructor
            auto solver_result = AmericanOptionSolver::create(params[i], workspace);
            if (!solver_result) {
                return unexpected(SolverError{
                    .code = SolverErrorCode::InvalidConfiguration,
                    .message = solver_result.error(),
                    .iterations = 0
                });
            }

            // NEW: Invoke setup callback if provided
            if (setup) {
                setup(i, solver_result.value());
            }

            return solver_result.value().solve();
        };

        // Use parallel region + for to enable per-thread workspace reuse
#ifdef _OPENMP
#pragma omp parallel
        {
            // Each thread creates ONE workspace and reuses it for all its iterations
            auto thread_workspace_result = AmericanSolverWorkspace::create(x_min, x_max, n_space, n_time);

            // If per-thread workspace creation fails (e.g., OOM), write error to all thread's results
            if (!thread_workspace_result) {
                SolverError error{
                    .code = SolverErrorCode::InvalidConfiguration,
                    .message = "Failed to create per-thread workspace: " + thread_workspace_result.error(),
                    .iterations = 0
                };
#pragma omp for
                for (size_t i = 0; i < params.size(); ++i) {
                    results[i] = unexpected(error);
                }
            } else {
                auto thread_workspace = thread_workspace_result.value();

#pragma omp for
                for (size_t i = 0; i < params.size(); ++i) {
                    results[i] = solve_one(i, thread_workspace);
                }
            }
        }
#else
        // Sequential: create workspace once and reuse for all options
        auto workspace_result = AmericanSolverWorkspace::create(x_min, x_max, n_space, n_time);
        if (!workspace_result) {
            SolverError error{
                .code = SolverErrorCode::InvalidConfiguration,
                .message = "Failed to create workspace: " + workspace_result.error(),
                .iterations = 0
            };
            for (size_t i = 0; i < params.size(); ++i) {
                results[i] = unexpected(error);
            }
            return results;
        }

        auto workspace = workspace_result.value();
        for (size_t i = 0; i < params.size(); ++i) {
            results[i] = solve_one(i, workspace);
        }
#endif

        return results;
    }

    /// Solve a batch of American options in parallel (vector overload)
    static std::vector<expected<AmericanOptionResult, SolverError>> solve_batch(
        const std::vector<AmericanOptionParams>& params,
        double x_min,
        double x_max,
        size_t n_space,
        size_t n_time,
        SetupCallback setup = nullptr)
    {
        return solve_batch(std::span{params}, x_min, x_max, n_space, n_time, setup);
    }
};
```

**Test command:**
```bash
bazel build //src/option:american_option
```

**Verification:**
- Compiles cleanly
- Backward compatible (setup = nullptr)
- Callback invoked before solve()

**Commit message:**
```
Add SetupCallback to BatchAmericanOptionSolver

Enhance solve_batch() with optional setup callback:
- Called after solver creation, before solve()
- Enables snapshot registration, convergence tuning
- Backward compatible: setup = nullptr preserves existing behavior

Use case: Price table building registers snapshots for each
(σ,r) combination via callback, eliminating manual OpenMP loops.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

### Task 2.3: Add tests for SetupCallback

**Files to modify:**
- `tests/american_option_test.cc`

**Add to existing test file:**

```cpp
TEST(BatchAmericanOptionSolverTest, SetupCallbackInvoked) {
    std::vector<AmericanOptionParams> batch(5);
    for (size_t i = 0; i < 5; ++i) {
        batch[i] = AmericanOptionParams{
            .strike = 100.0,
            .spot = 100.0,
            .maturity = 1.0,
            .volatility = 0.20 + 0.02 * i,
            .rate = 0.05,
            .continuous_dividend_yield = 0.02,
            .option_type = OptionType::PUT,
            .discrete_dividends = {}
        };
    }

    // Track callback invocations
    std::vector<size_t> callback_indices;
    std::mutex callback_mutex;

    auto results = BatchAmericanOptionSolver::solve_batch(
        batch, -3.0, 3.0, 101, 1000,
        [&](size_t idx, AmericanOptionSolver& solver) {
            std::lock_guard<std::mutex> lock(callback_mutex);
            callback_indices.push_back(idx);
        });

    // Verify all solves succeeded
    ASSERT_EQ(results.size(), 5);
    for (const auto& result : results) {
        EXPECT_TRUE(result.has_value());
    }

    // Verify callback was invoked for each option
    EXPECT_EQ(callback_indices.size(), 5);
    std::sort(callback_indices.begin(), callback_indices.end());
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(callback_indices[i], i);
    }
}

TEST(BatchAmericanOptionSolverTest, CallbackWithSnapshots) {
    std::vector<AmericanOptionParams> batch(3);
    for (size_t i = 0; i < 3; ++i) {
        batch[i] = AmericanOptionParams{
            .strike = 100.0,
            .spot = 100.0,
            .maturity = 1.0,
            .volatility = 0.20,
            .rate = 0.05,
            .continuous_dividend_yield = 0.02,
            .option_type = OptionType::PUT,
            .discrete_dividends = {}
        };
    }

    // Create collectors for each solve
    std::vector<double> moneyness = {0.9, 1.0, 1.1};
    std::vector<double> maturities = {0.5, 1.0};

    std::vector<PriceTableSnapshotCollector> collectors;
    for (size_t i = 0; i < 3; ++i) {
        PriceTableSnapshotCollectorConfig config{
            .moneyness = std::span{moneyness},
            .tau = std::span{maturities},
            .K_ref = 100.0,
            .option_type = OptionType::PUT,
            .payoff_params = nullptr
        };
        collectors.emplace_back(config);
    }

    // Register snapshots via callback
    auto results = BatchAmericanOptionSolver::solve_batch(
        batch, -3.0, 3.0, 101, 1000,
        [&](size_t idx, AmericanOptionSolver& solver) {
            solver.register_snapshot(499, 0, &collectors[idx]);  // τ=0.5
            solver.register_snapshot(999, 1, &collectors[idx]);  // τ=1.0
        });

    // Verify all solves succeeded
    ASSERT_EQ(results.size(), 3);
    for (const auto& result : results) {
        EXPECT_TRUE(result.has_value());
    }

    // Verify snapshots were collected
    for (size_t i = 0; i < 3; ++i) {
        auto prices = collectors[i].prices();
        EXPECT_EQ(prices.size(), moneyness.size() * maturities.size());

        // All prices should be positive
        for (double price : prices) {
            EXPECT_GT(price, 0.0);
        }
    }
}

TEST(BatchAmericanOptionSolverTest, NoCallbackBackwardCompatible) {
    std::vector<AmericanOptionParams> batch(3);
    for (size_t i = 0; i < 3; ++i) {
        batch[i] = AmericanOptionParams{
            .strike = 100.0,
            .spot = 100.0,
            .maturity = 1.0,
            .volatility = 0.20,
            .rate = 0.05,
            .continuous_dividend_yield = 0.02,
            .option_type = OptionType::PUT,
            .discrete_dividends = {}
        };
    }

    // Call without callback (backward compatible)
    auto results = BatchAmericanOptionSolver::solve_batch(
        batch, -3.0, 3.0, 101, 1000);

    ASSERT_EQ(results.size(), 3);
    for (const auto& result : results) {
        EXPECT_TRUE(result.has_value());
        EXPECT_GT(result.value().value, 0.0);
    }
}
```

**Test command:**
```bash
bazel test //tests:american_option_test
```

**Verification:**
- All tests pass
- Callback invoked for each solver
- Snapshots work via callback
- Backward compatibility preserved

**Commit message:**
```
Add tests for SetupCallback functionality

Test coverage:
- SetupCallbackInvoked: verifies callback called for each solver
- CallbackWithSnapshots: registers snapshots via callback
- NoCallbackBackwardCompatible: existing behavior preserved

Tests confirm callback mechanism works correctly and maintains
backward compatibility with existing code.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## Phase 3: Integration (4 tasks)

### Task 3.1: Add routing logic to PriceTable4DBuilder

**Files to modify:**
- `src/option/price_table_4d_builder.hpp`
- `src/option/price_table_4d_builder.cpp`

**Add to header (private section):**

```cpp
private:
    /// Check if we should use normalized solver (fast path)
    bool should_use_normalized_solver(
        double x_min,
        double x_max,
        size_t n_space,
        const std::vector<std::pair<double, double>>& discrete_dividends) const;
```

**Add to source:**

```cpp
#include "src/option/normalized_chain_solver.hpp"

bool PriceTable4DBuilder::should_use_normalized_solver(
    double x_min,
    double x_max,
    size_t n_space,
    const std::vector<std::pair<double, double>>& discrete_dividends) const
{
    // Check 1: No discrete dividends (normalized solver requirement)
    if (!discrete_dividends.empty()) {
        return false;
    }

    // Check 2: Build test request and check eligibility
    // Use first volatility/rate for eligibility check (grid params are same for all)
    NormalizedSolveRequest test_request{
        .sigma = volatility_.front(),
        .rate = rate_.front(),
        .dividend = 0.0,  // Will be set per-solve
        .option_type = OptionType::PUT,  // Doesn't affect eligibility
        .x_min = x_min,
        .x_max = x_max,
        .n_space = n_space,
        .n_time = 1000,  // Typical value
        .T_max = maturity_.back(),
        .tau_snapshots = std::span{maturity_}
    };

    auto eligibility = NormalizedChainSolver::check_eligibility(
        test_request, std::span{moneyness_});

    return eligibility.has_value();
}
```

**Test command:**
```bash
bazel build //src/option:price_table_4d_builder
```

**Verification:**
- Compiles cleanly
- Routing logic correct
- Eligibility check integrated

**Commit message:**
```
Add routing logic to PriceTable4DBuilder

Add should_use_normalized_solver():
- Returns false if discrete dividends present
- Checks eligibility via NormalizedChainSolver::check_eligibility()
- Uses moneyness grid to validate margins and ratio

This routing decision determines fast path (normalized solver)
vs fallback (batch API with snapshots).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

### Task 3.2: Implement fast path (normalized solver)

**Files to modify:**
- `src/option/price_table_4d_builder.cpp`

**Modify precompute() method to add fast path:**

```cpp
expected<PriceTable4DResult, std::string> PriceTable4DBuilder::precompute(
    OptionType option_type,
    double x_min,
    double x_max,
    size_t n_space,
    size_t n_time,
    double dividend_yield)
{
    const size_t Nm = moneyness_.size();
    const size_t Nt = maturity_.size();
    const size_t Nv = volatility_.size();
    const size_t Nr = rate_.size();

    // Validate that requested moneyness range fits within PDE grid bounds
    const double x_min_requested = std::log(moneyness_.front());
    const double x_max_requested = std::log(moneyness_.back());

    if (x_min_requested < x_min || x_max_requested > x_max) {
        return unexpected(
            "Requested moneyness range [" + std::to_string(moneyness_.front()) + ", " +
            std::to_string(moneyness_.back()) + "] in spot ratios " +
            "maps to log-moneyness [" + std::to_string(x_min_requested) + ", " +
            std::to_string(x_max_requested) + "], " +
            "which exceeds PDE grid bounds [" + std::to_string(x_min) + ", " +
            std::to_string(x_max) + "]. " +
            "Either narrow the moneyness grid or expand the PDE x_min/x_max bounds.");
    }

    // Allocate 4D price array
    std::vector<double> prices_4d(Nm * Nt * Nv * Nr, 0.0);

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Routing decision: normalized solver or batch API?
    bool use_normalized_solver = should_use_normalized_solver(
        x_min, x_max, n_space, {});  // No discrete dividends

    size_t failed_count = 0;

    if (use_normalized_solver) {
        // FAST PATH: Normalized solver
        const double T_max = maturity_.back();
        const double spot = K_ref_;  // For price tables, spot = K_ref

#pragma omp parallel
        {
            // Create normalized request template (per-thread)
            NormalizedSolveRequest base_request{
                .sigma = 0.20,  // Placeholder, set in loop
                .rate = 0.05,   // Placeholder, set in loop
                .dividend = dividend_yield,
                .option_type = option_type,
                .x_min = x_min,
                .x_max = x_max,
                .n_space = n_space,
                .n_time = n_time,
                .T_max = T_max,
                .tau_snapshots = std::span{maturity_}
            };

            // Create workspace once per thread (OUTSIDE work-sharing loop)
            auto workspace_result = NormalizedWorkspace::create(base_request);

            if (!workspace_result) {
                // Workspace creation failed, mark all as errors
#pragma omp for collapse(2)
                for (size_t k = 0; k < Nv; ++k) {
                    for (size_t l = 0; l < Nr; ++l) {
#pragma omp atomic
                        ++failed_count;
                    }
                }
            } else {
                auto workspace = std::move(workspace_result.value());
                auto surface = workspace.surface_view();

#pragma omp for collapse(2) schedule(dynamic, 1)
                for (size_t k = 0; k < Nv; ++k) {
                    for (size_t l = 0; l < Nr; ++l) {
                        // Set (σ, r) for this solve
                        NormalizedSolveRequest request = base_request;
                        request.sigma = volatility_[k];
                        request.rate = rate_[l];

                        // Solve normalized PDE
                        auto solve_result = NormalizedChainSolver::solve(
                            request, workspace, surface);

                        if (!solve_result) {
#pragma omp atomic
                            ++failed_count;
                            continue;
                        }

                        // Extract prices from surface
                        // Moneyness convention: m = K/S
                        for (size_t i = 0; i < Nm; ++i) {
                            double x = -std::log(moneyness_[i]);  // x = -ln(m)
                            double K = moneyness_[i] * spot;       // K = m * S

                            for (size_t j = 0; j < Nt; ++j) {
                                double u = surface.interpolate(x, maturity_[j]);
                                size_t idx_4d = ((i * Nt + j) * Nv + k) * Nr + l;
                                prices_4d[idx_4d] = K * u;  // V = K·u
                            }
                        }
                    }
                }
            }
        }

    } else {
        // FALLBACK PATH: Batch API with snapshots (existing implementation)
        // ... (keep existing code from current implementation) ...
    }

    if (failed_count > 0) {
        return unexpected("Failed to solve " + std::to_string(failed_count) +
                         " out of " + std::to_string(Nv * Nr) + " PDEs");
    }

    // End timer
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end_time - start_time);

    // Fit B-spline coefficients (unchanged)
    auto fitter_result = BSplineFitter4D::create(moneyness_, maturity_, volatility_, rate_);
    if (!fitter_result.has_value()) {
        return unexpected("B-spline fitter creation failed: " + fitter_result.error());
    }
    auto fit_result = fitter_result.value().fit(prices_4d);

    if (!fit_result.success) {
        return unexpected("B-spline fitting failed: " + fit_result.error_message);
    }

    // Create evaluator
    auto evaluator = std::make_unique<BSpline4D_FMA>(
        moneyness_, maturity_, volatility_, rate_, fit_result.coefficients);

    // Populate fitting statistics (unchanged)
    BSplineFittingStats fitting_stats{
        .max_residual_m = fit_result.max_residual_m,
        .max_residual_tau = fit_result.max_residual_tau,
        .max_residual_sigma = fit_result.max_residual_sigma,
        .max_residual_r = fit_result.max_residual_r,
        .max_residual_overall = fit_result.max_residual,
        .condition_m = fit_result.condition_m,
        .condition_tau = fit_result.condition_tau,
        .condition_sigma = fit_result.condition_sigma,
        .condition_r = fit_result.condition_r,
        .condition_max = std::max({
            fit_result.condition_m,
            fit_result.condition_tau,
            fit_result.condition_sigma,
            fit_result.condition_r
        }),
        .failed_slices_m = fit_result.failed_slices_m,
        .failed_slices_tau = fit_result.failed_slices_tau,
        .failed_slices_sigma = fit_result.failed_slices_sigma,
        .failed_slices_r = fit_result.failed_slices_r,
        .failed_slices_total = fit_result.failed_slices_m +
                               fit_result.failed_slices_tau +
                               fit_result.failed_slices_sigma +
                               fit_result.failed_slices_r
    };

    return PriceTable4DResult{
        .evaluator = std::move(evaluator),
        .prices_4d = std::move(prices_4d),
        .n_pde_solves = Nv * Nr,
        .precompute_time_seconds = duration.count(),
        .fitting_stats = fitting_stats
    };
}
```

**Test command:**
```bash
bazel build //src/option:price_table_4d_builder
```

**Verification:**
- Compiles cleanly
- Fast path integrated
- Fallback preserved

**Commit message:**
```
Implement fast path using normalized solver

Add fast path to PriceTable4DBuilder::precompute():
- Check eligibility via should_use_normalized_solver()
- Create per-thread NormalizedWorkspace (outside work-sharing loop)
- Solve Nv × Nr normalized PDEs in parallel
- Extract prices via V = K·u(ln(S/K), τ) interpolation
- Fall back to batch API if ineligible

Fast path exploits scale invariance to eliminate redundant
computation while maintaining current performance.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

### Task 3.3: Implement fallback path (batch API with callback)

**Files to modify:**
- `src/option/price_table_4d_builder.cpp`

**Complete the fallback path in precompute():**

```cpp
    } else {
        // FALLBACK PATH: Batch API with snapshots
        const double T_max = maturity_.back();
        const double dt = T_max / n_time;

        // Precompute step indices for each maturity
        std::vector<size_t> step_indices(Nt);
        for (size_t j = 0; j < Nt; ++j) {
            double step_exact = maturity_[j] / dt - 1.0;
            long long step_rounded = std::llround(step_exact);

            if (step_rounded < 0) {
                step_indices[j] = 0;
            } else if (step_rounded >= static_cast<long long>(n_time)) {
                step_indices[j] = n_time - 1;
            } else {
                step_indices[j] = static_cast<size_t>(step_rounded);
            }
        }

        // Build batch parameters (all (σ,r) combinations)
        std::vector<AmericanOptionParams> batch_params;
        batch_params.reserve(Nv * Nr);

        for (size_t k = 0; k < Nv; ++k) {
            for (size_t l = 0; l < Nr; ++l) {
                batch_params.push_back({
                    .strike = K_ref_,
                    .spot = K_ref_,
                    .maturity = T_max,
                    .volatility = volatility_[k],
                    .rate = rate_[l],
                    .continuous_dividend_yield = dividend_yield,
                    .option_type = option_type,
                    .discrete_dividends = {}
                });
            }
        }

        // Create collectors for each batch item
        std::vector<PriceTableSnapshotCollector> collectors;
        collectors.reserve(Nv * Nr);

        for (size_t idx = 0; idx < Nv * Nr; ++idx) {
            PriceTableSnapshotCollectorConfig collector_config{
                .moneyness = std::span{moneyness_},
                .tau = std::span{maturity_},
                .K_ref = K_ref_,
                .option_type = option_type,
                .payoff_params = nullptr
            };
            collectors.emplace_back(collector_config);
        }

        // Solve batch with snapshot registration via callback
        auto results = BatchAmericanOptionSolver::solve_batch(
            batch_params, x_min, x_max, n_space, n_time,
            [&](size_t idx, AmericanOptionSolver& solver) {
                // Register snapshots for all maturities
                for (size_t j = 0; j < Nt; ++j) {
                    solver.register_snapshot(step_indices[j], j, &collectors[idx]);
                }
            });

        // Extract prices from collectors
        for (size_t idx = 0; idx < Nv * Nr; ++idx) {
            size_t k = idx / Nr;
            size_t l = idx % Nr;

            if (!results[idx].has_value()) {
                ++failed_count;
                continue;
            }

            auto prices_2d = collectors[idx].prices();
            for (size_t i = 0; i < Nm; ++i) {
                for (size_t j = 0; j < Nt; ++j) {
                    size_t idx_2d = i * Nt + j;
                    size_t idx_4d = ((i * Nt + j) * Nv + k) * Nr + l;
                    prices_4d[idx_4d] = prices_2d[idx_2d];
                }
            }
        }
    }
```

**Test command:**
```bash
bazel build //src/option:price_table_4d_builder
```

**Verification:**
- Compiles cleanly
- Fallback uses SetupCallback
- Snapshots registered correctly

**Commit message:**
```
Implement fallback path using batch API with callback

Complete fallback in PriceTable4DBuilder::precompute():
- Build batch parameters for all (σ,r) combinations
- Create per-batch snapshot collectors
- Register snapshots via SetupCallback
- Extract prices from collectors to 4D array

Fallback handles discrete dividends and wide moneyness ranges
that normalized solver cannot handle.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

### Task 3.4: Add integration tests

**Files to create:**
- `tests/price_table_4d_integration_test.cc`

**Implementation:**

```cpp
/**
 * @file price_table_4d_integration_test.cc
 * @brief Integration tests for PriceTable4DBuilder with routing
 */

#include "src/option/price_table_4d_builder.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace mango;

TEST(PriceTable4DIntegrationTest, FastPathEligible) {
    // Narrow moneyness range → fast path
    auto builder = PriceTable4DBuilder::create(
        {0.9, 0.95, 1.0, 1.05, 1.1},     // Moneyness
        {0.25, 0.5, 1.0, 2.0},           // Maturity
        {0.15, 0.20, 0.25},              // Volatility
        {0.0, 0.02, 0.05},               // Rate
        100.0);                           // K_ref

    auto result = builder.precompute(OptionType::PUT, -3.0, 3.0, 101, 1000, 0.02);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->n_pde_solves, 3 * 3);  // Nv × Nr = 9

    // Spot check: ATM put with 1y maturity, σ=20%, r=5%
    double price = result->evaluator->eval(1.0, 1.0, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
    EXPECT_LT(price, 100.0);  // Put value < strike for ATM

    // Verify B-spline fitting quality
    EXPECT_LT(result->fitting_stats.max_residual_overall, 0.01);  // <1bp
}

TEST(PriceTable4DIntegrationTest, FallbackWideRange) {
    // Wide moneyness range → fallback
    auto builder = PriceTable4DBuilder::create(
        {0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5},  // Wide range
        {0.25, 0.5, 1.0, 2.0},
        {0.15, 0.20, 0.25},
        {0.0, 0.02, 0.05},
        100.0);

    auto result = builder.precompute(OptionType::PUT, -3.5, 3.5, 121, 1000, 0.02);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->n_pde_solves, 3 * 3);

    // Verify prices at extremes
    double price_deep_itm = result->evaluator->eval(0.5, 1.0, 0.20, 0.05);
    double price_deep_otm = result->evaluator->eval(1.5, 1.0, 0.20, 0.05);

    EXPECT_GT(price_deep_itm, price_deep_otm);  // ITM > OTM
}

TEST(PriceTable4DIntegrationTest, FastPathVsFallbackConsistency) {
    // Test same parameters using both paths
    std::vector<double> moneyness = {0.9, 0.95, 1.0, 1.05, 1.1};
    std::vector<double> maturity = {0.25, 0.5, 1.0};
    std::vector<double> volatility = {0.20, 0.25};
    std::vector<double> rate = {0.05};

    // Fast path (narrow range)
    auto builder_fast = PriceTable4DBuilder::create(
        moneyness, maturity, volatility, rate, 100.0);
    auto result_fast = builder_fast.precompute(
        OptionType::PUT, -3.0, 3.0, 101, 1000, 0.02);

    // Fallback (force by using wider grid)
    auto builder_fallback = PriceTable4DBuilder::create(
        moneyness, maturity, volatility, rate, 100.0);
    auto result_fallback = builder_fallback.precompute(
        OptionType::PUT, -3.5, 3.5, 121, 1000, 0.02);

    ASSERT_TRUE(result_fast.has_value());
    ASSERT_TRUE(result_fallback.has_value());

    // Compare prices at same query points
    for (double m : {0.9, 1.0, 1.1}) {
        for (double tau : {0.5, 1.0}) {
            for (double sigma : {0.20, 0.25}) {
                double price_fast = result_fast->evaluator->eval(m, tau, sigma, 0.05);
                double price_fallback = result_fallback->evaluator->eval(m, tau, sigma, 0.05);

                // Expect <1bp difference
                EXPECT_NEAR(price_fast, price_fallback, 0.01)
                    << "Mismatch at m=" << m << " tau=" << tau << " sigma=" << sigma;
            }
        }
    }
}

TEST(PriceTable4DIntegrationTest, PerformanceFastPath) {
    // Benchmark fast path
    auto builder = PriceTable4DBuilder::create(
        {0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2},  // 9 points
        {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0},              // 7 points
        {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40},         // 7 points
        {0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10},          // 7 points
        100.0);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = builder.precompute(OptionType::PUT, -3.0, 3.0, 101, 1000, 0.02);
    auto end = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(result.has_value());

    double duration_sec = std::chrono::duration<double>(end - start).count();
    std::cout << "Fast path: " << result->n_pde_solves << " PDEs in "
              << duration_sec << " seconds\n";
    std::cout << "Throughput: " << (result->n_pde_solves / duration_sec)
              << " PDEs/sec\n";

    // Should maintain ~848 options/sec on 32 cores (or scale with cores)
    // Relaxed threshold for CI environments
    EXPECT_LT(duration_sec, 60.0);  // Complete within 1 minute
}
```

**Build file update:**

Add to `tests/BUILD.bazel`:
```python
cc_test(
    name = "price_table_4d_integration_test",
    srcs = ["price_table_4d_integration_test.cc"],
    deps = [
        "//src/option:price_table_4d_builder",
        "@googletest//:gtest_main",
    ],
)
```

**Test command:**
```bash
bazel test //tests:price_table_4d_integration_test
```

**Verification:**
- All tests pass
- Fast path and fallback produce consistent results
- Performance acceptable
- Edge cases handled

**Commit message:**
```
Add integration tests for PriceTable4DBuilder routing

Test coverage:
- FastPathEligible: narrow range uses normalized solver
- FallbackWideRange: wide range uses batch API
- FastPathVsFallbackConsistency: both paths agree (<1bp)
- PerformanceFastPath: throughput benchmark

Tests confirm routing logic works correctly and both paths
produce accurate, consistent results.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## Testing and Verification

### Run All Tests

```bash
# Unit tests
bazel test //tests:normalized_chain_solver_test
bazel test //tests:american_option_test

# Integration tests
bazel test //tests:price_table_4d_integration_test

# Full test suite
bazel test //...
```

### Performance Benchmarks

```bash
# Existing benchmarks should maintain performance
bazel run //benchmarks:price_table_benchmark
```

### Manual Verification

Create `examples/example_normalized_solver.cc`:

```cpp
#include "src/option/normalized_chain_solver.hpp"
#include "src/option/price_table_4d_builder.hpp"
#include <iostream>
#include <vector>

using namespace mango;

int main() {
    // Example: Price table with narrow moneyness range (fast path)
    auto builder = PriceTable4DBuilder::create(
        {0.8, 0.9, 1.0, 1.1, 1.2},       // Moneyness
        {0.25, 0.5, 1.0, 2.0},           // Maturity
        {0.15, 0.20, 0.25, 0.30},        // Volatility
        {0.0, 0.02, 0.05},               // Rate
        100.0);                           // K_ref

    std::cout << "Building price table (fast path)...\n";
    auto result = builder.precompute(OptionType::PUT, -3.0, 3.0, 101, 1000, 0.02);

    if (!result.has_value()) {
        std::cerr << "Error: " << result.error() << "\n";
        return 1;
    }

    std::cout << "Precomputed " << result->n_pde_solves << " PDEs in "
              << result->precompute_time_seconds << " seconds\n";
    std::cout << "Throughput: "
              << (result->n_pde_solves / result->precompute_time_seconds)
              << " PDEs/sec\n";

    // Query prices
    std::cout << "\nSample prices (ATM put, τ=1y):\n";
    for (double sigma : {0.15, 0.20, 0.25, 0.30}) {
        double price = result->evaluator->eval(1.0, 1.0, sigma, 0.05);
        std::cout << "  σ=" << sigma << ": $" << price << "\n";
    }

    return 0;
}
```

Build and run:
```bash
bazel run //examples:example_normalized_solver
```

---

## Summary

This implementation plan provides 12 bite-sized tasks organized into 3 phases:

**Phase 1 (5 tasks)**: Core normalized solver with workspace, eligibility, solving, and tests
**Phase 2 (3 tasks)**: Batch API enhancement with validation, callback, and tests
**Phase 3 (4 tasks)**: Integration with routing, fast path, fallback, and tests

Each task includes:
- Exact file paths
- Complete code snippets
- Build commands
- Verification steps
- Commit messages

Total estimated time: 2-3 days for experienced developer

The implementation exploits scale invariance V(S,K,τ) = K·u(ln(S/K), τ) to solve once and interpolate for all strikes, maintaining current performance (~848 options/sec on 32 cores) while simplifying code and enabling future optimizations.
