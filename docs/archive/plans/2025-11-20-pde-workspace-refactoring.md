# PDE Workspace Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Separate Grid (persistent metadata + solution), Workspace (temporary PMR buffers), and PDESolver (compute-only) to eliminate buffer ownership confusion and enable derivative caching.

**Architecture:** Three-component separation with clear ownership: Grid owns solution + metadata (shared_ptr), Workspace provides named spans to PMR buffers (caller-managed), PDESolver uses spans for zero-copy computation.

**Tech Stack:** C++23, std::span, std::pmr::vector, CRTP templates, std::expected

**Related Issues:** #206 (buffer confusion), #203 (duplicate arrays), #207 (caching blocked), #208 (variant error)

---

## Phase 1: Create Grid Class

### Task 1.1: Write Grid class header with tests

**Files:**
- Create: `src/pde/core/grid_with_solution.hpp`
- Create: `tests/grid_with_solution_test.cc`
- Modify: `src/pde/core/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing test**

Create `tests/grid_with_solution_test.cc`:

```cpp
#include "src/pde/core/grid_with_solution.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"
#include <gtest/gtest.h>

namespace mango {

TEST(GridWithSolutionTest, CreateUniformGrid) {
    // Setup
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_spec.has_value());

    TimeDomain time{.t_start = 0.0, .t_end = 1.0, .n_steps = 1000};

    // Act
    auto grid_result = GridWithSolution<double>::create(grid_spec.value(), time);

    // Assert
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value();

    EXPECT_EQ(grid->n_space(), 101);
    EXPECT_EQ(grid->n_time(), 1000);
    EXPECT_DOUBLE_EQ(grid->dt(), 0.001);
    EXPECT_EQ(grid->x().size(), 101);
}

TEST(GridWithSolutionTest, CreateSinhSpacedGrid) {
    // Setup
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0);
    ASSERT_TRUE(grid_spec.has_value());

    TimeDomain time{.t_start = 0.0, .t_end = 1.0, .n_steps = 500};

    // Act
    auto grid_result = GridWithSolution<double>::create(grid_spec.value(), time);

    // Assert
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value();

    EXPECT_EQ(grid->n_space(), 201);
    EXPECT_FALSE(grid->spacing().is_uniform());
}

TEST(GridWithSolutionTest, SolutionBufferSize) {
    // Setup
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    TimeDomain time{.t_start = 0.0, .t_end = 1.0, .n_steps = 1000};

    // Act
    auto grid = GridWithSolution<double>::create(grid_spec.value(), time).value();

    // Assert - stores last 2 time steps for theta
    EXPECT_EQ(grid->solution().size(), 101);
    EXPECT_EQ(grid->solution_prev().size(), 101);

    // Should be writable
    grid->solution()[0] = 1.23;
    EXPECT_DOUBLE_EQ(grid->solution()[0], 1.23);
}

TEST(GridWithSolutionTest, GridSpacingAccess) {
    // Setup
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    TimeDomain time{.t_start = 0.0, .t_end = 1.0, .n_steps = 1000};

    // Act
    auto grid = GridWithSolution<double>::create(grid_spec.value(), time).value();

    // Assert - returns const reference (no variant copy!)
    const auto& spacing = grid->spacing();
    EXPECT_TRUE(spacing.is_uniform());
    EXPECT_DOUBLE_EQ(spacing.spacing(), 0.01);  // (1.0 - 0.0) / 100 intervals
}

} // namespace mango
```

**Step 2: Run test to verify it fails**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "grid_with_solution_test",
    srcs = ["grid_with_solution_test.cc"],
    deps = [
        "//src/pde/core:grid_with_solution",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

Add to `src/pde/core/BUILD.bazel`:

```python
cc_library(
    name = "grid_with_solution",
    hdrs = ["grid_with_solution.hpp"],
    deps = [
        ":grid",
        ":time_domain",
    ],
    visibility = ["//visibility:public"],
)
```

Run: `bazel test //tests:grid_with_solution_test --test_output=all`

Expected: FAIL with "grid_with_solution.hpp: No such file or directory"

**Step 3: Write minimal implementation**

Create `src/pde/core/grid_with_solution.hpp`:

```cpp
#ifndef MANGO_GRID_WITH_SOLUTION_HPP
#define MANGO_GRID_WITH_SOLUTION_HPP

#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"
#include <expected>
#include <memory>
#include <vector>
#include <span>
#include <optional>

namespace mango {

/**
 * Grid with solution storage for PDE solvers.
 *
 * Owns:
 * - Spatial grid (x, spacing, knot_vector)
 * - Time domain (dt, n_steps)
 * - Solution storage: last 2 time steps (for theta computation)
 *
 * Lifetime: Outlives PDESolver and Workspace (passed via shared_ptr)
 */
template<typename T>
class GridWithSolution {
public:
    /**
     * Factory method with validation.
     *
     * @param grid_spec Spatial grid specification
     * @param time Time domain configuration
     * @return shared_ptr to Grid on success, error message on failure
     */
    static std::expected<std::shared_ptr<GridWithSolution>, std::string>
    create(const GridSpec<T>& grid_spec, const TimeDomain& time) {
        // Generate grid buffer
        auto grid_buffer = grid_spec.generate();

        // Create GridSpacing
        auto spacing_result = GridSpacing<T>::create(
            grid_buffer.span(),
            grid_buffer.span()  // dx computed from grid
        );

        if (!spacing_result.has_value()) {
            return std::unexpected(spacing_result.error());
        }

        // Allocate solution storage (2 × n_space for theta)
        size_t n_space = grid_buffer.span().size();
        std::vector<T> solution(2 * n_space);

        // Create Grid (cannot use make_shared with private constructor)
        auto grid = std::shared_ptr<GridWithSolution>(
            new GridWithSolution(
                std::move(grid_buffer),
                spacing_result.value(),
                time,
                std::move(solution)
            )
        );

        return grid;
    }

    // Spatial grid accessors
    std::span<const T> x() const { return grid_buffer_.span(); }
    size_t n_space() const { return grid_buffer_.span().size(); }

    // Grid spacing (returns const& - no variant copy!)
    const GridSpacing<T>& spacing() const { return spacing_; }

    // Time domain accessors
    const TimeDomain& time() const { return time_; }
    size_t n_time() const { return time_.n_steps(); }
    T dt() const { return time_.dt(); }

    // Solution buffers (size: n_space each)
    std::span<T> solution() {
        return std::span{solution_.data(), n_space()};
    }

    std::span<const T> solution() const {
        return std::span{solution_.data(), n_space()};
    }

    std::span<T> solution_prev() {
        return std::span{solution_.data() + n_space(), n_space()};
    }

    std::span<const T> solution_prev() const {
        return std::span{solution_.data() + n_space(), n_space()};
    }

    // Knot vector for B-spline interpolation (lazy-computed, cached)
    std::span<const T> knot_vector() const {
        if (!knot_cache_.has_value()) {
            knot_cache_ = clamped_knots_cubic(x());
        }
        return *knot_cache_;
    }

private:
    // Private constructor (use create() factory)
    GridWithSolution(GridBuffer<T> grid_buffer,
                     GridSpacing<T> spacing,
                     TimeDomain time,
                     std::vector<T> solution)
        : grid_buffer_(std::move(grid_buffer))
        , spacing_(std::move(spacing))
        , time_(time)
        , solution_(std::move(solution))
    {}

    GridBuffer<T> grid_buffer_;           // Spatial grid points
    GridSpacing<T> spacing_;               // Uniform or non-uniform
    TimeDomain time_;                      // Time stepping info
    std::vector<T> solution_;              // [u_current | u_prev] (2 × n_space)
    mutable std::optional<std::vector<T>> knot_cache_;  // Lazy-computed
};

} // namespace mango

#endif // MANGO_GRID_WITH_SOLUTION_HPP
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:grid_with_solution_test --test_output=all`

Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add src/pde/core/grid_with_solution.hpp tests/grid_with_solution_test.cc src/pde/core/BUILD.bazel tests/BUILD.bazel
git commit -m "Add GridWithSolution class with solution storage

Stores spatial grid, time domain, and last 2 time steps (for theta).
Lazy-computes knot vector for B-spline interpolation.
Returns GridSpacing by const& to avoid variant copy (fixes #208).

Related: #209"
```

---

### Task 1.2: Add knot vector caching test

**Files:**
- Modify: `tests/grid_with_solution_test.cc`

**Step 1: Write the failing test**

Add to `tests/grid_with_solution_test.cc`:

```cpp
TEST(GridWithSolutionTest, KnotVectorCaching) {
    // Setup
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    TimeDomain time{.t_start = 0.0, .t_end = 1.0, .n_steps = 1000};
    auto grid = GridWithSolution<double>::create(grid_spec.value(), time).value();

    // Act - first call computes knot vector
    auto knots1 = grid->knot_vector();

    // Act - second call returns cached copy
    auto knots2 = grid->knot_vector();

    // Assert - same data pointer (cached)
    EXPECT_EQ(knots1.data(), knots2.data());

    // Assert - knot vector size is n + 4 for cubic B-splines
    EXPECT_EQ(knots1.size(), 101 + 4);

    // Assert - clamped endpoints (repeated 4 times)
    EXPECT_DOUBLE_EQ(knots1[0], 0.0);
    EXPECT_DOUBLE_EQ(knots1[1], 0.0);
    EXPECT_DOUBLE_EQ(knots1[2], 0.0);
    EXPECT_DOUBLE_EQ(knots1[3], 0.0);
    EXPECT_DOUBLE_EQ(knots1[101], 1.0);
    EXPECT_DOUBLE_EQ(knots1[102], 1.0);
    EXPECT_DOUBLE_EQ(knots1[103], 1.0);
    EXPECT_DOUBLE_EQ(knots1[104], 1.0);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:grid_with_solution_test --test_output=all`

Expected: FAIL with "clamped_knots_cubic was not declared in this scope"

**Step 3: Add missing include**

Modify `src/pde/core/grid_with_solution.hpp`:

```cpp
#include "src/bspline/bspline_utils.hpp"  // Add this line
```

Update `src/pde/core/BUILD.bazel`:

```python
cc_library(
    name = "grid_with_solution",
    hdrs = ["grid_with_solution.hpp"],
    deps = [
        ":grid",
        ":time_domain",
        "//src/bspline:bspline_utils",  # Add this
    ],
    visibility = ["//visibility:public"],
)
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:grid_with_solution_test --test_output=all`

Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add tests/grid_with_solution_test.cc src/pde/core/grid_with_solution.hpp src/pde/core/BUILD.bazel
git commit -m "Test knot vector caching in GridWithSolution

Verifies lazy computation and caching of B-spline knot vector.

Related: #209"
```

---

## Phase 2: Create Workspace Struct

### Task 2.1: Write Workspace struct with tests

**Files:**
- Create: `src/pde/core/pde_workspace_spans.hpp`
- Create: `tests/pde_workspace_spans_test.cc`
- Modify: `src/pde/core/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing test**

Create `tests/pde_workspace_spans_test.cc`:

```cpp
#include "src/pde/core/pde_workspace_spans.hpp"
#include <gtest/gtest.h>
#include <memory_resource>
#include <vector>

namespace mango {

TEST(WorkspaceSpansTest, RequiredSize) {
    // Act
    size_t size = PDEWorkspaceSpans::required_size(101);

    // Assert - needs 7 arrays: rhs, jacobian_diag, jacobian_upper, jacobian_lower, residual, delta_u, psi
    // Each array: 101 elements, rounded to 8-element SIMD boundary
    size_t n_padded = ((101 + 7) / 8) * 8;  // 104
    EXPECT_EQ(size, 7 * n_padded);
}

TEST(WorkspaceSpansTest, FromBuffer) {
    // Setup
    size_t n = 101;
    size_t buffer_size = PDEWorkspaceSpans::required_size(n);
    std::vector<double> buffer(buffer_size);

    // Act
    auto workspace = PDEWorkspaceSpans::from_buffer(buffer, n);

    // Assert - all spans have correct size
    EXPECT_EQ(workspace.rhs().size(), n);
    EXPECT_EQ(workspace.jacobian_diag().size(), n);
    EXPECT_EQ(workspace.jacobian_upper().size(), n);
    EXPECT_EQ(workspace.jacobian_lower().size(), n);
    EXPECT_EQ(workspace.residual().size(), n);
    EXPECT_EQ(workspace.delta_u().size(), n);
    EXPECT_EQ(workspace.psi().size(), n);
}

TEST(WorkspaceSpansTest, BufferLayout) {
    // Setup
    size_t n = 101;
    size_t buffer_size = PDEWorkspaceSpans::required_size(n);
    std::vector<double> buffer(buffer_size, 0.0);

    // Act
    auto workspace = PDEWorkspaceSpans::from_buffer(buffer, n);

    // Modify via spans
    workspace.rhs()[0] = 1.0;
    workspace.jacobian_diag()[0] = 2.0;
    workspace.psi()[50] = 3.0;

    // Assert - modifications visible in original buffer
    EXPECT_DOUBLE_EQ(buffer[0], 1.0);  // rhs starts at offset 0
    // (exact offsets depend on padding, just verify non-zero)
    bool found_2 = false;
    bool found_3 = false;
    for (auto val : buffer) {
        if (val == 2.0) found_2 = true;
        if (val == 3.0) found_3 = true;
    }
    EXPECT_TRUE(found_2);
    EXPECT_TRUE(found_3);
}

TEST(WorkspaceSpansTest, PMRIntegration) {
    // Setup - use PMR pool
    std::pmr::synchronized_pool_resource pool;
    size_t n = 101;
    size_t buffer_size = PDEWorkspaceSpans::required_size(n);

    std::pmr::vector<double> pmr_buffer(buffer_size, &pool);

    // Act
    auto workspace = PDEWorkspaceSpans::from_buffer(pmr_buffer, n);

    // Assert - spans work with PMR-backed buffer
    workspace.rhs()[0] = 42.0;
    EXPECT_DOUBLE_EQ(pmr_buffer[0], 42.0);
}

} // namespace mango
```

**Step 2: Run test to verify it fails**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "pde_workspace_spans_test",
    srcs = ["pde_workspace_spans_test.cc"],
    deps = [
        "//src/pde/core:pde_workspace_spans",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

Add to `src/pde/core/BUILD.bazel`:

```python
cc_library(
    name = "pde_workspace_spans",
    hdrs = ["pde_workspace_spans.hpp"],
    visibility = ["//visibility:public"],
)
```

Run: `bazel test //tests:pde_workspace_spans_test --test_output=all`

Expected: FAIL with "pde_workspace_spans.hpp: No such file or directory"

**Step 3: Write minimal implementation**

Create `src/pde/core/pde_workspace_spans.hpp`:

```cpp
#ifndef MANGO_PDE_WORKSPACE_SPANS_HPP
#define MANGO_PDE_WORKSPACE_SPANS_HPP

#include <span>
#include <cstddef>

namespace mango {

/**
 * Named spans into a caller-provided workspace buffer.
 *
 * Provides zero-copy access to temporary working arrays for PDE solver.
 * Caller manages buffer lifetime and allocation strategy (PMR pool, arena, default).
 *
 * Layout: [rhs | jacobian_diag | jacobian_upper | jacobian_lower | residual | delta_u | psi]
 * Each array padded to 8-element boundary for SIMD operations.
 */
struct PDEWorkspaceSpans {
    /**
     * Calculate required buffer size for n grid points.
     *
     * @param n Number of spatial grid points
     * @return Buffer size in doubles (includes SIMD padding)
     */
    static size_t required_size(size_t n) {
        // Pad to 8-element boundary for AVX-512
        size_t n_padded = ((n + 7) / 8) * 8;

        // 7 arrays: rhs, jacobian_diag, jacobian_upper, jacobian_lower, residual, delta_u, psi
        return 7 * n_padded;
    }

    /**
     * Create workspace spans from caller-provided buffer.
     *
     * @param buffer Backing storage (must be at least required_size(n) elements)
     * @param n Number of spatial grid points
     * @return Workspace with named spans into buffer
     */
    static PDEWorkspaceSpans from_buffer(std::span<double> buffer, size_t n) {
        size_t n_padded = ((n + 7) / 8) * 8;

        PDEWorkspaceSpans workspace;
        workspace.n_ = n;

        // Slice buffer into named spans
        size_t offset = 0;
        workspace.rhs_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.jacobian_diag_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.jacobian_upper_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.jacobian_lower_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.residual_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.delta_u_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.psi_ = buffer.subspan(offset, n);

        return workspace;
    }

    // Accessors
    std::span<double> rhs() { return rhs_; }
    std::span<double> jacobian_diag() { return jacobian_diag_; }
    std::span<double> jacobian_upper() { return jacobian_upper_; }
    std::span<double> jacobian_lower() { return jacobian_lower_; }
    std::span<double> residual() { return residual_; }
    std::span<double> delta_u() { return delta_u_; }
    std::span<double> psi() { return psi_; }

    std::span<const double> rhs() const { return rhs_; }
    std::span<const double> jacobian_diag() const { return jacobian_diag_; }
    std::span<const double> jacobian_upper() const { return jacobian_upper_; }
    std::span<const double> jacobian_lower() const { return jacobian_lower_; }
    std::span<const double> residual() const { return residual_; }
    std::span<const double> delta_u() const { return delta_u_; }
    std::span<const double> psi() const { return psi_; }

    size_t size() const { return n_; }

private:
    size_t n_;
    std::span<double> rhs_;
    std::span<double> jacobian_diag_;
    std::span<double> jacobian_upper_;
    std::span<double> jacobian_lower_;
    std::span<double> residual_;
    std::span<double> delta_u_;
    std::span<double> psi_;
};

} // namespace mango

#endif // MANGO_PDE_WORKSPACE_SPANS_HPP
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:pde_workspace_spans_test --test_output=all`

Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/pde/core/pde_workspace_spans.hpp tests/pde_workspace_spans_test.cc src/pde/core/BUILD.bazel tests/BUILD.bazel
git commit -m "Add PDEWorkspaceSpans for named workspace access

Provides zero-copy spans into caller-managed buffer.
Supports PMR for efficient batch operations.
SIMD-padded arrays for vectorized PDE operations.

Related: #209"
```

---

## Phase 3: Update PDESolver (No Obstacle CRTP Yet)

### Task 3.1: Add Grid + Workspace constructor to PDESolver

**Files:**
- Modify: `src/pde/core/pde_solver.hpp`
- Modify: `tests/pde_solver_test.cc`

**Step 1: Write the failing test**

Add to `tests/pde_solver_test.cc`:

```cpp
TEST_F(PDESolverTest, ConstructorWithGridAndWorkspace) {
    // Setup - create grid
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    TimeDomain time{.t_start = 0.0, .t_end = 1.0, .n_steps = 100};
    auto grid = GridWithSolution<double>::create(grid_spec.value(), time).value();

    // Setup - create workspace buffer
    size_t workspace_size = PDEWorkspaceSpans::required_size(101);
    std::vector<double> workspace_buffer(workspace_size);
    auto workspace = PDEWorkspaceSpans::from_buffer(workspace_buffer, 101);

    // Setup - create test solver
    auto solver = TestPDESolver(grid, workspace);

    // Assert - solver initialized correctly
    EXPECT_EQ(solver.n_space(), 101);
    EXPECT_EQ(solver.n_time(), 100);

    // Initialize with simple payoff
    solver.initialize([](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(M_PI * x[i]);
        }
    });

    // Solve
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    // Assert - final solution stored in grid
    auto solution = solver.grid()->solution();
    EXPECT_EQ(solution.size(), 101);
    EXPECT_GT(solution[50], 0.0);  // Some non-zero value
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:pde_solver_test --test_output=all`

Expected: FAIL with "no matching constructor for PDESolver"

**Step 3: Add Grid + Workspace constructor to PDESolver**

Modify `src/pde/core/pde_solver.hpp`:

```cpp
template<typename Derived>
class PDESolver {
public:
    // ... existing constructors ...

    /**
     * Constructor with Grid and Workspace (new refactored design).
     *
     * @param grid Persistent grid with solution storage (shared_ptr, outlives solver)
     * @param workspace Named spans to caller-managed temporary buffers
     * @param obstacle Optional obstacle constraint callback
     */
    PDESolver(std::shared_ptr<GridWithSolution<double>> grid,
              PDEWorkspaceSpans workspace,
              std::optional<ObstacleCallback> obstacle = std::nullopt)
        : grid_with_solution_(grid)
        , workspace_spans_(workspace)
        , time_(grid->time())
        , config_()
        , obstacle_(std::move(obstacle))
        , n_(grid->n_space())
    {
        // Allocate internal working buffers (size: n_space each)
        u_current_.resize(n_);
        u_old_.resize(n_);
    }

    // ... rest of class ...

    /// Access grid (for post-processing)
    std::shared_ptr<GridWithSolution<double>> grid() const {
        return grid_with_solution_;
    }

    size_t n_space() const { return n_; }
    size_t n_time() const { return time_.n_steps(); }

private:
    // New members for Grid + Workspace design
    std::shared_ptr<GridWithSolution<double>> grid_with_solution_;
    PDEWorkspaceSpans workspace_spans_;

    // Internal working buffers (used during solve())
    std::vector<double> u_current_;
    std::vector<double> u_old_;

    // ... existing members ...
};
```

Update `src/pde/core/BUILD.bazel`:

```python
cc_library(
    name = "pde_solver",
    hdrs = ["pde_solver.hpp"],
    deps = [
        ":grid",
        ":grid_with_solution",     # Add this
        ":pde_workspace_spans",    # Add this
        ":time_domain",
        "//src/support:error_types",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:pde_solver_test --test_output=all`

Expected: PASS (new test + existing tests)

**Step 5: Commit**

```bash
git add src/pde/core/pde_solver.hpp tests/pde_solver_test.cc src/pde/core/BUILD.bazel
git commit -m "Add Grid + Workspace constructor to PDESolver

New constructor accepts GridWithSolution and PDEWorkspaceSpans.
Solver uses internal u_current_/u_old_ during solve().
Final 2 steps written to Grid after solve completes.

Related: #209"
```

---

### Task 3.2: Update solve() to write final 2 steps to Grid

**Files:**
- Modify: `src/pde/core/pde_solver.hpp`
- Modify: `tests/pde_solver_test.cc`

**Step 1: Write the failing test**

Add to `tests/pde_solver_test.cc`:

```cpp
TEST_F(PDESolverTest, SolveWritesFinalTwoStepsToGrid) {
    // Setup
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    TimeDomain time{.t_start = 0.0, .t_end = 1.0, .n_steps = 100};
    auto grid = GridWithSolution<double>::create(grid_spec.value(), time).value();

    size_t workspace_size = PDEWorkspaceSpans::required_size(101);
    std::vector<double> workspace_buffer(workspace_size);
    auto workspace = PDEWorkspaceSpans::from_buffer(workspace_buffer, 101);

    auto solver = TestPDESolver(grid, workspace);

    // Initialize with payoff
    solver.initialize([](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(M_PI * x[i]);
        }
    });

    // Act - solve
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    // Assert - grid contains final 2 time steps
    auto u_final = grid->solution();
    auto u_prev = grid->solution_prev();

    EXPECT_EQ(u_final.size(), 101);
    EXPECT_EQ(u_prev.size(), 101);

    // u_final and u_prev should be different (evolved solution)
    double diff = 0.0;
    for (size_t i = 0; i < 101; ++i) {
        diff += std::abs(u_final[i] - u_prev[i]);
    }
    EXPECT_GT(diff, 0.0);  // Not identical
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:pde_solver_test --test_output=all`

Expected: FAIL - grid solution buffers are zero (not written)

**Step 3: Update solve() to write final 2 steps**

Modify `src/pde/core/pde_solver.hpp` in the `solve()` method:

```cpp
std::expected<void, SolverError> solve() {
    // ... existing solve logic using u_current_ and u_old_ ...

    for (size_t step = 0; step < time_.n_steps(); ++step) {
        // Swap buffers before updating
        std::swap(u_current_, u_old_);

        // Stage 1: Trapezoidal rule
        // ... existing code ...

        // Stage 2: BDF2
        // ... existing code ...

        // Snapshot callback (if set)
        if (snapshot_callback_) {
            auto dest = (*snapshot_callback_)(step + 1, t_next);
            if (!dest.empty()) {
                std::copy(u_current_.begin(), u_current_.end(), dest.begin());
            }
        }
    }

    // NEW: Write final 2 time steps to Grid
    if (grid_with_solution_) {
        auto grid_current = grid_with_solution_->solution();
        auto grid_prev = grid_with_solution_->solution_prev();

        std::copy(u_current_.begin(), u_current_.end(), grid_current.begin());
        std::copy(u_old_.begin(), u_old_.end(), grid_prev.begin());
    }

    return {};
}
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:pde_solver_test --test_output=all`

Expected: PASS (grid contains final 2 steps)

**Step 5: Commit**

```bash
git add src/pde/core/pde_solver.hpp tests/pde_solver_test.cc
git commit -m "Write final 2 time steps to Grid after solve()

Enables theta computation via finite differences.
Grid stores u(t_final) and u(t_final - dt).

Related: #209"
```

---

## Phase 4: Update AmericanOptionSolver

### Task 4.1: Update AmericanPutSolver to use Grid + Workspace

**Files:**
- Modify: `src/option/american_pde_solver.hpp`
- Modify: `tests/american_option_test.cc`

**Step 1: Write the failing test**

Add to `tests/american_option_test.cc`:

```cpp
TEST_F(AmericanOptionTest, SolverWithGridWorkspaceDesign) {
    // Setup - pricing params
    PricingParams params{
        .spot = 100.0,
        .strike = 100.0,
        .maturity = 1.0,
        .rate = 0.05,
        .dividend = 0.02,
        .type = OptionType::PUT,
        .volatility = 0.20
    };

    // Setup - create grid
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0).value();
    TimeDomain time{.t_start = 0.0, .t_end = params.maturity, .n_steps = 1000};
    auto grid = GridWithSolution<double>::create(grid_spec, time).value();

    // Setup - create workspace
    size_t workspace_size = PDEWorkspaceSpans::required_size(101);
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> workspace_buffer(workspace_size, &pool);
    auto workspace = PDEWorkspaceSpans::from_buffer(workspace_buffer, 101);

    // Act - create and solve
    AmericanPutSolver solver(params, grid, workspace);
    solver.initialize(AmericanPutSolver::payoff);

    auto result = solver.solve();

    // Assert
    ASSERT_TRUE(result.has_value());

    // Grid contains final solution
    auto solution = grid->solution();
    EXPECT_GT(solution[50], 0.0);  // ATM put has time value

    // Grid contains previous step (for theta)
    auto solution_prev = grid->solution_prev();
    EXPECT_GT(solution_prev[50], 0.0);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:american_option_test --test_output=all`

Expected: FAIL with "no matching constructor for AmericanPutSolver"

**Step 3: Update AmericanPutSolver constructor**

Modify `src/option/american_pde_solver.hpp`:

```cpp
class AmericanPutSolver : public PDESolver<AmericanPutSolver> {
public:
    // NEW: Grid + Workspace constructor
    AmericanPutSolver(const PricingParams& params,
                      std::shared_ptr<GridWithSolution<double>> grid,
                      PDEWorkspaceSpans workspace)
        : PDESolver<AmericanPutSolver>(
              grid,
              workspace,
              create_obstacle(params.strike))  // Obstacle callback
        , params_(params)
        , left_bc_(create_left_bc(params))
        , right_bc_(create_right_bc(params))
        , spatial_op_(create_spatial_op(params, grid->spacing()))
    {}

    // ... existing methods ...

private:
    // Helper to create obstacle callback
    static std::optional<ObstacleCallback> create_obstacle(double K) {
        return [K](double t, std::span<const double> x, std::span<double> psi) {
            for (size_t i = 0; i < x.size(); ++i) {
                double S = K * std::exp(x[i]);
                psi[i] = std::max(K - S, 0.0);
            }
        };
    }

    // ... existing members ...
};
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:american_option_test --test_output=all`

Expected: PASS (solver works with Grid + Workspace)

**Step 5: Commit**

```bash
git add src/option/american_pde_solver.hpp tests/american_option_test.cc
git commit -m "Update AmericanPutSolver to use Grid + Workspace

New constructor accepts GridWithSolution and PDEWorkspaceSpans.
Obstacle callback created from strike price.

Related: #209"
```

---

### Task 4.2: Implement theta computation using Grid

**Files:**
- Modify: `src/option/american_option.cpp`
- Modify: `tests/american_option_test.cc`

**Step 1: Write the failing test**

Add to `tests/american_option_test.cc`:

```cpp
TEST_F(AmericanOptionTest, ThetaComputationWithGridDesign) {
    // Setup
    PricingParams params{
        .spot = 100.0,
        .strike = 100.0,
        .maturity = 1.0,
        .rate = 0.05,
        .dividend = 0.02,
        .type = OptionType::PUT,
        .volatility = 0.20
    };

    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0).value();
    TimeDomain time{.t_start = 0.0, .t_end = params.maturity, .n_steps = 1000};
    auto grid = GridWithSolution<double>::create(grid_spec, time).value();

    size_t workspace_size = PDEWorkspaceSpans::required_size(101);
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> workspace_buffer(workspace_size, &pool);
    auto workspace = PDEWorkspaceSpans::from_buffer(workspace_buffer, 101);

    AmericanPutSolver solver(params, grid, workspace);
    solver.initialize(AmericanPutSolver::payoff);
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    // Act - compute theta
    double theta = solver.compute_theta();

    // Assert - theta should be negative (time decay)
    EXPECT_LT(theta, 0.0);

    // Assert - theta magnitude should be reasonable (not zero stub)
    EXPECT_LT(theta, -0.001);  // At least 0.1 cent per day decay
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:american_option_test --test_output=all`

Expected: FAIL - theta is 0.0 (stub implementation)

**Step 3: Implement compute_theta() using Grid**

Modify `src/option/american_option.cpp`:

```cpp
double AmericanOptionSolver::compute_theta() const {
    if (!solved_) {
        return 0.0;
    }

    // Get Grid from solver
    auto grid = get_grid();  // Assumes solver stores grid_ member
    if (!grid) {
        return 0.0;
    }

    // Theta = -∂V/∂t ≈ -(V(t) - V(t-dt)) / dt
    // Use backward finite difference (first-order accurate)

    auto u_current = grid->solution();   // V(t_final)
    auto u_prev = grid->solution_prev(); // V(t_final - dt)
    double dt = grid->dt();

    // Find current spot in grid
    double current_moneyness = std::log(params_.spot / params_.strike);

    // Interpolate both time steps at current spot
    double V_current = interpolate_solution(current_moneyness, grid->x(), u_current);
    double V_prev = interpolate_solution(current_moneyness, grid->x(), u_prev);

    // Finite difference: ∂V/∂t
    double dV_dt = (V_current - V_prev) / dt;

    // Theta convention: negative of time decay
    return -dV_dt;
}
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:american_option_test --test_output=all`

Expected: PASS (theta is negative and non-zero)

**Step 5: Commit**

```bash
git add src/option/american_option.cpp tests/american_option_test.cc
git commit -m "Implement theta computation using Grid storage

Uses backward finite difference: (V(t) - V(t-dt)) / dt
Requires Grid to store last 2 time steps.
First-order accurate (O(dt) error).

Fixes #208 stub implementation
Related: #209"
```

---

## Phase 5: Eliminate AmericanSolverWorkspace

### Task 5.1: Remove AmericanSolverWorkspace class

**Files:**
- Delete: `src/option/american_solver_workspace.hpp`
- Delete: `src/option/american_solver_workspace.cpp`
- Delete: `tests/american_solver_workspace_test.cc`
- Modify: `src/option/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Verify all usages are removed**

Run: `git grep "AmericanSolverWorkspace" src/ tests/`

Expected: Only finds definitions in files to be deleted

**Step 2: Delete files**

```bash
git rm src/option/american_solver_workspace.hpp
git rm src/option/american_solver_workspace.cpp
git rm tests/american_solver_workspace_test.cc
```

**Step 3: Update BUILD files**

Remove from `src/option/BUILD.bazel`:

```python
cc_library(
    name = "american_solver_workspace",
    # DELETE THIS ENTIRE TARGET
)
```

Remove from `tests/BUILD.bazel`:

```python
cc_test(
    name = "american_solver_workspace_test",
    # DELETE THIS ENTIRE TARGET
)
```

**Step 4: Run all tests to verify nothing breaks**

Run: `bazel test //... --test_output=errors`

Expected: All tests PASS (no references to deleted class)

**Step 5: Commit**

```bash
git add -u  # Stage deletions
git commit -m "Remove AmericanSolverWorkspace (replaced by Grid + Workspace)

Eliminated bundled workspace class in favor of:
- GridWithSolution: persistent metadata + solution storage
- PDEWorkspaceSpans: caller-managed temporary buffers

Simplifies ownership and enables PMR reuse.

Fixes #206, #203
Related: #209"
```

---

## Phase 6: Obstacle CRTP Pattern (Optional Optimization)

### Task 6.1: Add NoObstacle CRTP class

**Files:**
- Create: `src/pde/core/obstacle_policies.hpp`
- Create: `tests/obstacle_policies_test.cc`
- Modify: `src/pde/core/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the failing test**

Create `tests/obstacle_policies_test.cc`:

```cpp
#include "src/pde/core/obstacle_policies.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace mango {
namespace obstacles {

TEST(NoObstacleTest, ZeroOverhead) {
    // Setup
    NoObstacle<double> obstacle;
    std::vector<double> x = {-1.0, 0.0, 1.0};
    std::vector<double> psi(3);

    // Act
    obstacle.apply(0.5, x, psi);

    // Assert - no-op (psi unchanged)
    EXPECT_EQ(psi[0], 0.0);
    EXPECT_EQ(psi[1], 0.0);
    EXPECT_EQ(psi[2], 0.0);

    // Assert - has tag type
    using Tag = decltype(obstacle)::tag;
    EXPECT_TRUE((std::is_same_v<Tag, no_obstacle_tag>));
}

TEST(AmericanPutObstacleTest, ComputesPayoff) {
    // Setup
    double K = 100.0;
    AmericanPutObstacle<double> obstacle(K);

    std::vector<double> x = {-0.5, 0.0, 0.5};  // log-moneyness
    std::vector<double> psi(3);

    // Act
    obstacle.apply(0.5, x, psi);

    // Assert - psi(x) = max(K - S, 0) = max(K - K*exp(x), 0)
    // x = -0.5: S = 100*exp(-0.5) ≈ 60.65, psi ≈ 39.35
    EXPECT_NEAR(psi[0], 100.0 - 100.0 * std::exp(-0.5), 1e-10);

    // x = 0.0: S = 100, psi = 0
    EXPECT_NEAR(psi[1], 0.0, 1e-10);

    // x = 0.5: S = 100*exp(0.5) ≈ 164.87, psi = 0
    EXPECT_NEAR(psi[2], 0.0, 1e-10);
}

TEST(AmericanCallObstacleTest, ComputesPayoff) {
    // Setup
    double K = 100.0;
    AmericanCallObstacle<double> obstacle(K);

    std::vector<double> x = {-0.5, 0.0, 0.5};
    std::vector<double> psi(3);

    // Act
    obstacle.apply(0.5, x, psi);

    // Assert - psi(x) = max(S - K, 0) = max(K*exp(x) - K, 0)
    // x = -0.5: S ≈ 60.65, psi = 0
    EXPECT_NEAR(psi[0], 0.0, 1e-10);

    // x = 0.0: S = 100, psi = 0
    EXPECT_NEAR(psi[1], 0.0, 1e-10);

    // x = 0.5: S ≈ 164.87, psi ≈ 64.87
    EXPECT_NEAR(psi[2], 100.0 * std::exp(0.5) - 100.0, 1e-10);
}

} // namespace obstacles
} // namespace mango
```

**Step 2: Run test to verify it fails**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "obstacle_policies_test",
    srcs = ["obstacle_policies_test.cc"],
    deps = [
        "//src/pde/core:obstacle_policies",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

Add to `src/pde/core/BUILD.bazel`:

```python
cc_library(
    name = "obstacle_policies",
    hdrs = ["obstacle_policies.hpp"],
    visibility = ["//visibility:public"],
)
```

Run: `bazel test //tests:obstacle_policies_test --test_output=all`

Expected: FAIL with "obstacle_policies.hpp: No such file or directory"

**Step 3: Write minimal implementation**

Create `src/pde/core/obstacle_policies.hpp`:

```cpp
#ifndef MANGO_OBSTACLE_POLICIES_HPP
#define MANGO_OBSTACLE_POLICIES_HPP

#include <span>
#include <cmath>
#include <algorithm>

namespace mango {
namespace obstacles {

// Tag types for compile-time dispatch
struct no_obstacle_tag {};
struct american_put_tag {};
struct american_call_tag {};

/**
 * NoObstacle: Zero-overhead empty implementation.
 *
 * Used for European options where no obstacle constraint is needed.
 * Compiler eliminates apply() call via if constexpr.
 */
template<typename T>
class NoObstacle {
public:
    using tag = no_obstacle_tag;

    void apply([[maybe_unused]] T t,
               [[maybe_unused]] std::span<const T> x,
               [[maybe_unused]] std::span<T> psi) const {
        // No-op (compiler eliminates this entirely with if constexpr)
    }
};

/**
 * AmericanPutObstacle: Stateless CRTP implementation.
 *
 * Computes put obstacle: psi(x) = max(K - S, 0) = max(K - K*exp(x), 0)
 * where x is log-moneyness (x = ln(S/K)).
 */
template<typename T>
class AmericanPutObstacle {
public:
    using tag = american_put_tag;

    explicit AmericanPutObstacle(T strike) : K_(strike) {}

    void apply([[maybe_unused]] T t,
               std::span<const T> x,
               std::span<T> psi) const {
        for (size_t i = 0; i < x.size(); ++i) {
            T S = K_ * std::exp(x[i]);
            psi[i] = std::max(K_ - S, T(0));
        }
    }

private:
    T K_;
};

/**
 * AmericanCallObstacle: Stateless CRTP implementation.
 *
 * Computes call obstacle: psi(x) = max(S - K, 0) = max(K*exp(x) - K, 0)
 */
template<typename T>
class AmericanCallObstacle {
public:
    using tag = american_call_tag;

    explicit AmericanCallObstacle(T strike) : K_(strike) {}

    void apply([[maybe_unused]] T t,
               std::span<const T> x,
               std::span<T> psi) const {
        for (size_t i = 0; i < x.size(); ++i) {
            T S = K_ * std::exp(x[i]);
            psi[i] = std::max(S - K_, T(0));
        }
    }

private:
    T K_;
};

} // namespace obstacles
} // namespace mango

#endif // MANGO_OBSTACLE_POLICIES_HPP
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:obstacle_policies_test --test_output=all`

Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add src/pde/core/obstacle_policies.hpp tests/obstacle_policies_test.cc src/pde/core/BUILD.bazel tests/BUILD.bazel
git commit -m "Add CRTP obstacle policies (NoObstacle, American)

Zero-overhead NoObstacle for European options.
Stateless AmericanPutObstacle and AmericanCallObstacle.
Tag dispatch enables compile-time branch elimination.

Related: #209"
```

---

### Task 6.2: Update PDESolver to use Obstacle CRTP

**Files:**
- Modify: `src/pde/core/pde_solver.hpp`
- Modify: `tests/pde_solver_test.cc`

**Step 1: Write the failing test**

Add to `tests/pde_solver_test.cc`:

```cpp
TEST_F(PDESolverTest, ObstacleCRTPCompileCheck) {
    // This test verifies the CRTP template compiles with different obstacle policies
    // We're not implementing European options, just verifying the infrastructure works

    // Verify NoObstacle compiles (zero-overhead path)
    using NoObstacleType = obstacles::NoObstacle<double>;
    static_assert(std::is_same_v<NoObstacleType::tag, obstacles::no_obstacle_tag>);

    // Verify AmericanPutObstacle compiles
    using PutObstacleType = obstacles::AmericanPutObstacle<double>;
    static_assert(std::is_same_v<PutObstacleType::tag, obstacles::american_put_tag>);

    // Verify AmericanCallObstacle compiles
    using CallObstacleType = obstacles::AmericanCallObstacle<double>;
    static_assert(std::is_same_v<CallObstacleType::tag, obstacles::american_call_tag>);

    SUCCEED();  // Just verify template instantiation works
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:pde_solver_test --test_output=all`

Expected: FAIL with "PDESolver template parameter mismatch"

**Step 3: Update PDESolver template**

Modify `src/pde/core/pde_solver.hpp`:

```cpp
template<typename Derived, typename ObstaclePolicy = obstacles::NoObstacle<double>>
class PDESolver {
public:
    PDESolver(std::shared_ptr<GridWithSolution<double>> grid,
              PDEWorkspaceSpans workspace,
              ObstaclePolicy obstacle = ObstaclePolicy{})
        : grid_with_solution_(std::move(grid))
        , workspace_spans_(std::move(workspace))
        , obstacle_(std::move(obstacle))
        , time_(grid_with_solution_->time())
        , config_()
        , n_(grid_with_solution_->n_space())
    {
        u_current_.resize(n_);
        u_old_.resize(n_);
    }

    std::expected<void, SolverError> solve() {
        // ... time stepping loop ...

        for (size_t step = 0; step < time_.n_steps(); ++step) {
            // Newton iteration
            for (size_t iter = 0; iter < max_iter; ++iter) {
                // ... Newton solve ...

                // Apply obstacle constraint (compile-time dispatch)
                if constexpr (!std::is_same_v<typename ObstaclePolicy::tag,
                                               obstacles::no_obstacle_tag>) {
                    obstacle_.apply(t_current, grid_with_solution_->x(), workspace_spans_.psi());

                    // Project solution: u = max(u, psi)
                    auto u_span = u_current_;
                    auto psi_span = workspace_spans_.psi();
                    for (size_t i = 0; i < u_span.size(); ++i) {
                        u_span[i] = std::max(u_span[i], psi_span[i]);
                    }
                }
            }
        }

        return {};
    }

private:
    ObstaclePolicy obstacle_;
    // ... other members ...
};
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:pde_solver_test --test_output=all`

Expected: PASS (CRTP obstacle works)

**Step 5: Commit**

```bash
git add src/pde/core/pde_solver.hpp tests/pde_solver_test.cc
git commit -m "Add Obstacle CRTP template parameter to PDESolver

if constexpr eliminates obstacle branch for European options.
No virtual call overhead for American options.
Consistent CRTP pattern across all solver components.

Related: #209"
```

---

## Phase 7: Update Documentation and Cleanup

### Task 7.1: Update CLAUDE.md with new architecture

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add new architecture section**

Add after "American Option API Simplification" section in `CLAUDE.md`:

```markdown
### PDE Workspace Refactoring (Issue #209)

The PDE solver architecture separates three concerns for clear ownership:

**1. GridWithSolution: Persistent metadata + solution storage**

```cpp
auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0).value();
TimeDomain time{.t_start = 0.0, .t_end = 1.0, .n_steps = 1000};
auto grid = GridWithSolution<double>::create(grid_spec, time).value();

// Grid owns:
// - Spatial grid (x, spacing, knot_vector)
// - Time domain (dt, n_steps)
// - Solution storage: last 2 time steps (for theta)
```

**2. PDEWorkspaceSpans: Caller-managed temporary buffers**

```cpp
// Caller controls allocation strategy (PMR pool, arena, default)
std::pmr::synchronized_pool_resource pool;
size_t workspace_size = PDEWorkspaceSpans::required_size(101);
std::pmr::vector<double> buffer(workspace_size, &pool);

auto workspace = PDEWorkspaceSpans::from_buffer(buffer, 101);

// Workspace provides named spans (zero-copy):
// - rhs, jacobian_diag, jacobian_upper, jacobian_lower
// - residual, delta_u, psi
```

**3. PDESolver: Compute-only (no memory ownership)**

```cpp
AmericanPutSolver solver(params, grid, workspace);
solver.initialize(AmericanPutSolver::payoff);
auto result = solver.solve();

// After solve, grid contains:
// - Final solution: grid->solution()
// - Previous step: grid->solution_prev() (for theta)
```

**Benefits:**

- **Memory efficiency:** 500× reduction (808 KB → 1.6 KB per solve)
- **Workspace reuse:** Single PMR pool for batch IV operations
- **Clear ownership:** Grid (persistent), Workspace (temporary), Solver (compute)
- **Derivative caching:** Grid returns spacing by const& (no variant copy, fixes #208)
- **Theta computation:** Grid stores last 2 steps for finite differences

**Obstacle CRTP Pattern:**

```cpp
// NoObstacle provides zero-overhead path (if constexpr eliminates branch)
// Used as default template parameter - future European options would use this

// American put: stateless obstacle policy
class AmericanPutSolver
    : public PDESolver<AmericanPutSolver, AmericanPutObstacle<double>> {};

// American call: stateless obstacle policy
class AmericanCallSolver
    : public PDESolver<AmericanCallSolver, AmericanCallObstacle<double>> {};
```

See Issue #209 for complete design details.
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "Document PDE workspace refactoring in CLAUDE.md

Explains GridWithSolution, PDEWorkspaceSpans, and PDESolver separation.
Includes usage examples and benefits.

Related: #209"
```

---

### Task 7.2: Close related issues

**Files:**
- None (GitHub issue updates)

**Step 1: Close Issue #206**

```bash
gh issue close 206 --comment "Fixed by PR #XXX (PDE workspace refactoring).

Grid now owns solution buffer with clear layout:
- grid->solution(): u(t_final)
- grid->solution_prev(): u(t_final - dt)

No more pointer confusion."
```

**Step 2: Close Issue #203**

```bash
gh issue close 203 --comment "Fixed by PR #XXX (PDE workspace refactoring).

Single source of truth:
- GridWithSolution: persistent solution storage
- PDEWorkspaceSpans: temporary working arrays
- PDESolver: uses spans (no ownership)

No duplicate arrays."
```

**Step 3: Close Issue #207**

```bash
gh issue close 207 --comment "Fixed by PR #XXX (PDE workspace refactoring).

Grid outlives solver and returns GridSpacing by const& (no variant copy).
Derivative caching now works (see Issue #208 implementation)."
```

**Step 4: Update Issue #208**

```bash
gh issue comment 208 --body "Unblocked by PR #XXX (PDE workspace refactoring).

Grid design solves variant copy issue:
- grid->spacing() returns const& (no copy)
- Grid outlives solver (stable lifetime)
- Ready to implement derivative caching

Next: Add first_derivatives_ and second_derivatives_ to GridWithSolution."
```

**Step 5: Close Issue #209**

```bash
gh issue close 209 --comment "Implemented in PR #XXX.

Complete refactoring:
✅ GridWithSolution (persistent metadata + solution)
✅ PDEWorkspaceSpans (caller-managed temporary buffers)
✅ PDESolver (compute-only, uses spans)
✅ Obstacle CRTP pattern (zero overhead for European)
✅ Theta computation (last 2 time steps in Grid)
✅ 500× memory reduction (808 KB → 1.6 KB)

All tests passing. Documentation updated."
```

---

## Testing Strategy

### Unit Tests
- GridWithSolution: creation, accessors, knot caching
- PDEWorkspaceSpans: buffer slicing, PMR integration
- Obstacle policies: payoff computation, tag dispatch
- PDESolver: Grid + Workspace constructor, solve writes to Grid
- AmericanOptionSolver: theta computation, Greeks with Grid

### Integration Tests
- American option pricing (Put and Call) with new architecture
- Batch IV solving with PMR workspace reuse
- Greeks computation (delta, gamma, theta)
- NoObstacle infrastructure (compile checks only, not full European solver)

### Performance Tests
- Regression: ensure no slowdown vs current design
- Memory: verify 500× reduction for Grid storage
- Batch IV: compare malloc count (should be minimal with PMR)
- Obstacle overhead: verify minimal overhead with CRTP vs std::function

---

## Rollback Plan

If issues arise during implementation:

**Phase 1-2 (Grid + Workspace):**
- Revert commits
- Keep old PDESolver + AmericanSolverWorkspace

**Phase 3-4 (PDESolver + American):**
- Revert to Grid + Workspace commits
- Fix PDESolver integration before proceeding

**Phase 5 (Remove old workspace):**
- Can delay deletion (both designs coexist temporarily)
- Migrate tests incrementally

**Phase 6 (Obstacle CRTP):**
- Optional optimization (not critical path)
- Can skip if time-constrained

---

## Estimated Timeline

- **Phase 1:** Grid class (2 tasks) - 30 minutes
- **Phase 2:** Workspace struct (1 task) - 15 minutes
- **Phase 3:** PDESolver updates (2 tasks) - 45 minutes
- **Phase 4:** AmericanOptionSolver (2 tasks) - 30 minutes
- **Phase 5:** Cleanup (1 task) - 15 minutes
- **Phase 6:** Obstacle CRTP (2 tasks, optional) - 30 minutes
- **Phase 7:** Documentation (2 tasks) - 20 minutes

**Total:** ~3 hours (without obstacle CRTP), ~3.5 hours (with obstacle CRTP)

**Recommendation:** Implement Phases 1-5 first (core refactoring), then Phase 6 (optimization) if time permits.

---

Plan saved to: `docs/plans/2025-11-20-pde-workspace-refactoring.md`
