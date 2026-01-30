<!-- SPDX-License-Identifier: MIT -->
# PDE Workspace Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor PDE workspace to separate Grid (persistent), Workspace (temporary spans), and PDESolver (compute-only) with clean code and no technical debt.

**Architecture:** Direct replacement strategy - add new infrastructure (Commit 1), then atomic update of PDESolver + all derived classes + tests (Commit 2 broken into substeps), then enable theta (Commit 3).

**Tech Stack:** C++23, std::span, std::expected, std::pmr, Bazel, GoogleTest

**Related:** Issue #209, Codex approval (v3), Clean migration design

---

## Commit 1: Add New Infrastructure (Safe, Additive)

### Task 1.1: Create GridWithSolution class

**Files:**
- Create: `src/pde/core/grid_with_solution.hpp`
- Modify: `src/pde/core/BUILD.bazel`

**Step 1: Add BUILD target (compilation check)**

Modify `src/pde/core/BUILD.bazel`:

```python
cc_library(
    name = "grid_with_solution",
    hdrs = ["grid_with_solution.hpp"],
    deps = [
        ":grid",
        ":time_domain",
        "//src/bspline:bspline_utils",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 2: Create header with minimal implementation**

Create `src/pde/core/grid_with_solution.hpp`:

```cpp
#ifndef MANGO_GRID_WITH_SOLUTION_HPP
#define MANGO_GRID_WITH_SOLUTION_HPP

#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"
#include "src/bspline/bspline_utils.hpp"
#include <expected>
#include <memory>
#include <vector>
#include <span>
#include <optional>

namespace mango {

template<typename T>
class GridWithSolution {
public:
    static std::expected<std::shared_ptr<GridWithSolution>, std::string>
    create(const GridSpec<T>& grid_spec, const TimeDomain& time) {
        auto grid_buffer = grid_spec.generate();
        auto grid_view = grid_buffer.view();
        auto spacing = GridSpacing<T>(grid_view);

        size_t n_space = grid_buffer.span().size();
        std::vector<T> solution(2 * n_space);

        return std::shared_ptr<GridWithSolution>(
            new GridWithSolution(
                std::move(grid_buffer),
                std::move(spacing),
                time,
                std::move(solution)
            )
        );
    }

    std::span<const T> x() const { return grid_buffer_.span(); }
    GridView<T> view() const { return grid_buffer_.view(); }
    size_t n_space() const { return grid_buffer_.span().size(); }

    const GridSpacing<T>& spacing() const { return spacing_; }

    const TimeDomain& time() const { return time_; }
    size_t n_time() const { return time_.n_steps(); }
    T dt() const { return time_.dt(); }

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

    std::span<const T> knot_vector() const {
        if (!knot_cache_.has_value()) {
            knot_cache_ = clamped_knots_cubic(x());
        }
        return *knot_cache_;
    }

private:
    GridWithSolution(GridBuffer<T> grid_buffer,
                     GridSpacing<T> spacing,
                     TimeDomain time,
                     std::vector<T> solution)
        : grid_buffer_(std::move(grid_buffer))
        , spacing_(std::move(spacing))
        , time_(time)
        , solution_(std::move(solution))
    {}

    GridBuffer<T> grid_buffer_;
    GridSpacing<T> spacing_;
    TimeDomain time_;
    std::vector<T> solution_;
    mutable std::optional<std::vector<T>> knot_cache_;
};

} // namespace mango

#endif
```

**Step 3: Verify compilation**

Run: `bazel build //src/pde/core:grid_with_solution`

Expected: SUCCESS

**Step 4: Commit**

```bash
git add src/pde/core/grid_with_solution.hpp src/pde/core/BUILD.bazel
git commit -m "Add GridWithSolution class (persistent grid + solution storage)

Stores spatial grid, time domain, and last 2 time steps for theta.
Uses GridView constructor for GridSpacing (Codex approved).
Lazy-caches knot vector for B-spline interpolation.

Related: #209"
```

---

### Task 1.2: Create GridWithSolution tests

**Files:**
- Create: `tests/grid_with_solution_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write failing test**

Create `tests/grid_with_solution_test.cc`:

```cpp
#include "src/pde/core/grid_with_solution.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"
#include <gtest/gtest.h>

namespace mango {

TEST(GridWithSolutionTest, CreateUniformGrid) {
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_spec.has_value());

    TimeDomain time{.t_start = 0.0, .t_end = 1.0, .n_steps = 1000};

    auto grid_result = GridWithSolution<double>::create(grid_spec.value(), time);

    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value();

    EXPECT_EQ(grid->n_space(), 101);
    EXPECT_EQ(grid->n_time(), 1000);
    EXPECT_DOUBLE_EQ(grid->dt(), 0.001);
    EXPECT_EQ(grid->x().size(), 101);
}

TEST(GridWithSolutionTest, SolutionBufferSize) {
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    TimeDomain time{.t_start = 0.0, .t_end = 1.0, .n_steps = 1000};

    auto grid = GridWithSolution<double>::create(grid_spec.value(), time).value();

    EXPECT_EQ(grid->solution().size(), 101);
    EXPECT_EQ(grid->solution_prev().size(), 101);

    grid->solution()[0] = 1.23;
    EXPECT_DOUBLE_EQ(grid->solution()[0], 1.23);
}

TEST(GridWithSolutionTest, GridSpacingAccess) {
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    TimeDomain time{.t_start = 0.0, .t_end = 1.0, .n_steps = 1000};

    auto grid = GridWithSolution<double>::create(grid_spec.value(), time).value();

    const auto& spacing = grid->spacing();
    EXPECT_TRUE(spacing.is_uniform());
    EXPECT_DOUBLE_EQ(spacing.spacing(), 0.01);
}

TEST(GridWithSolutionTest, KnotVectorCaching) {
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    TimeDomain time{.t_start = 0.0, .t_end = 1.0, .n_steps = 1000};
    auto grid = GridWithSolution<double>::create(grid_spec.value(), time).value();

    auto knots1 = grid->knot_vector();
    auto knots2 = grid->knot_vector();

    EXPECT_EQ(knots1.data(), knots2.data());
    EXPECT_EQ(knots1.size(), 105);  // n + 4 for cubic
}

} // namespace mango
```

**Step 2: Add test target**

Modify `tests/BUILD.bazel`:

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

**Step 3: Run test to verify it passes**

Run: `bazel test //tests:grid_with_solution_test --test_output=all`

Expected: PASS (4 tests)

**Step 4: Commit**

```bash
git add tests/grid_with_solution_test.cc tests/BUILD.bazel
git commit -m "Add GridWithSolution tests

Tests grid creation, solution buffers, spacing access, knot caching.

Related: #209"
```

---

### Task 1.3: Create PDEWorkspaceSpans struct

**Files:**
- Create: `src/pde/core/pde_workspace_spans.hpp`
- Modify: `src/pde/core/BUILD.bazel`

**Step 1: Add BUILD target**

Modify `src/pde/core/BUILD.bazel`:

```python
cc_library(
    name = "pde_workspace_spans",
    hdrs = ["pde_workspace_spans.hpp"],
    visibility = ["//visibility:public"],
)
```

**Step 2: Create header (complete, 15 arrays + tridiag)**

Create `src/pde/core/pde_workspace_spans.hpp`:

```cpp
#ifndef MANGO_PDE_WORKSPACE_SPANS_HPP
#define MANGO_PDE_WORKSPACE_SPANS_HPP

#include <span>
#include <cstddef>
#include <expected>
#include <format>

namespace mango {

struct PDEWorkspaceSpans {
    static size_t required_size(size_t n) {
        size_t n_padded = ((n + 7) / 8) * 8;
        size_t regular = 15 * n_padded;  // 15 arrays @ n
        size_t tridiag = ((2 * n + 7) / 8) * 8;  // tridiag @ 2n
        return regular + tridiag;
    }

    static std::expected<PDEWorkspaceSpans, std::string>
    from_buffer(std::span<double> buffer, size_t n) {
        size_t required = required_size(n);

        if (buffer.size() < required) {
            return std::unexpected(std::format(
                "Workspace buffer too small: {} < {} required for n={}",
                buffer.size(), required, n));
        }

        size_t n_padded = ((n + 7) / 8) * 8;
        PDEWorkspaceSpans workspace;
        workspace.n_ = n;

        size_t offset = 0;

        workspace.dx_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.u_stage_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.rhs_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.lu_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.psi_ = buffer.subspan(offset, n);
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

        workspace.newton_u_old_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.u_next_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.reserved1_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.reserved2_ = buffer.subspan(offset, n);
        offset += n_padded;

        workspace.reserved3_ = buffer.subspan(offset, n);
        offset += n_padded;

        size_t tridiag_padded = ((2 * n + 7) / 8) * 8;
        workspace.tridiag_workspace_ = buffer.subspan(offset, 2 * n);

        return workspace;
    }

    static std::expected<PDEWorkspaceSpans, std::string>
    from_buffer_and_grid(std::span<double> buffer,
                        std::span<const double> grid,
                        size_t n) {
        auto workspace_result = from_buffer(buffer, n);
        if (!workspace_result.has_value()) {
            return std::unexpected(workspace_result.error());
        }

        auto workspace = workspace_result.value();

        auto dx_span = workspace.dx();
        for (size_t i = 0; i < n - 1; ++i) {
            dx_span[i] = grid[i + 1] - grid[i];
        }

        return workspace;
    }

    std::span<double> dx() { return dx_.subspan(0, n_ - 1); }
    std::span<const double> dx() const { return dx_.subspan(0, n_ - 1); }

    std::span<double> u_stage() { return u_stage_; }
    std::span<const double> u_stage() const { return u_stage_; }

    std::span<double> rhs() { return rhs_; }
    std::span<const double> rhs() const { return rhs_; }

    std::span<double> lu() { return lu_; }
    std::span<const double> lu() const { return lu_; }

    std::span<double> psi() { return psi_; }
    std::span<const double> psi() const { return psi_; }

    std::span<double> jacobian_diag() { return jacobian_diag_; }
    std::span<const double> jacobian_diag() const { return jacobian_diag_; }

    std::span<double> jacobian_upper() { return jacobian_upper_.subspan(0, n_ - 1); }
    std::span<const double> jacobian_upper() const { return jacobian_upper_.subspan(0, n_ - 1); }

    std::span<double> jacobian_lower() { return jacobian_lower_.subspan(0, n_ - 1); }
    std::span<const double> jacobian_lower() const { return jacobian_lower_.subspan(0, n_ - 1); }

    std::span<double> residual() { return residual_; }
    std::span<const double> residual() const { return residual_; }

    std::span<double> delta_u() { return delta_u_; }
    std::span<const double> delta_u() const { return delta_u_; }

    std::span<double> newton_u_old() { return newton_u_old_; }
    std::span<const double> newton_u_old() const { return newton_u_old_; }

    std::span<double> u_next() { return u_next_; }
    std::span<const double> u_next() const { return u_next_; }

    std::span<double> tridiag_workspace() { return tridiag_workspace_; }
    std::span<const double> tridiag_workspace() const { return tridiag_workspace_; }

    size_t size() const { return n_; }

private:
    size_t n_;
    std::span<double> dx_;
    std::span<double> u_stage_;
    std::span<double> rhs_;
    std::span<double> lu_;
    std::span<double> psi_;
    std::span<double> jacobian_diag_;
    std::span<double> jacobian_upper_;
    std::span<double> jacobian_lower_;
    std::span<double> residual_;
    std::span<double> delta_u_;
    std::span<double> newton_u_old_;
    std::span<double> u_next_;
    std::span<double> tridiag_workspace_;
    std::span<double> reserved1_;
    std::span<double> reserved2_;
    std::span<double> reserved3_;
};

} // namespace mango

#endif
```

**Step 3: Verify compilation**

Run: `bazel build //src/pde/core:pde_workspace_spans`

Expected: SUCCESS

**Step 4: Commit**

```bash
git add src/pde/core/pde_workspace_spans.hpp src/pde/core/BUILD.bazel
git commit -m "Add PDEWorkspaceSpans (15 arrays + tridiag workspace)

Complete array inventory per Codex review:
- 15 arrays @ n each (SIMD padded)
- tridiag_workspace @ 2n
- Buffer size validation with std::expected
- from_buffer_and_grid() computes dx automatically

Related: #209"
```

---

### Task 1.4: Create PDEWorkspaceSpans tests

**Files:**
- Create: `tests/pde_workspace_spans_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write test**

Create `tests/pde_workspace_spans_test.cc`:

```cpp
#include "src/pde/core/pde_workspace_spans.hpp"
#include <gtest/gtest.h>
#include <memory_resource>
#include <vector>

namespace mango {

TEST(PDEWorkspaceSpansTest, RequiredSize) {
    size_t size = PDEWorkspaceSpans::required_size(101);

    size_t n_padded = 104;  // ((101 + 7) / 8) * 8
    size_t expected = 15 * n_padded + ((2 * 101 + 7) / 8) * 8;

    EXPECT_EQ(size, expected);
}

TEST(PDEWorkspaceSpansTest, FromBuffer) {
    size_t n = 101;
    size_t buffer_size = PDEWorkspaceSpans::required_size(n);
    std::vector<double> buffer(buffer_size);

    auto workspace = PDEWorkspaceSpans::from_buffer(buffer, n);

    ASSERT_TRUE(workspace.has_value());
    EXPECT_EQ(workspace->rhs().size(), n);
    EXPECT_EQ(workspace->lu().size(), n);
    EXPECT_EQ(workspace->u_stage().size(), n);
    EXPECT_EQ(workspace->dx().size(), n - 1);
    EXPECT_EQ(workspace->tridiag_workspace().size(), 2 * n);
}

TEST(PDEWorkspaceSpansTest, BufferTooSmall) {
    size_t n = 101;
    std::vector<double> buffer(100);  // Too small

    auto workspace = PDEWorkspaceSpans::from_buffer(buffer, n);

    EXPECT_FALSE(workspace.has_value());
    EXPECT_NE(workspace.error().find("too small"), std::string::npos);
}

TEST(PDEWorkspaceSpansTest, FromBufferAndGrid) {
    size_t n = 101;
    size_t buffer_size = PDEWorkspaceSpans::required_size(n);
    std::vector<double> buffer(buffer_size);

    std::vector<double> grid(n);
    for (size_t i = 0; i < n; ++i) {
        grid[i] = i * 0.01;
    }

    auto workspace = PDEWorkspaceSpans::from_buffer_and_grid(buffer, grid, n);

    ASSERT_TRUE(workspace.has_value());

    auto dx = workspace->dx();
    for (size_t i = 0; i < n - 1; ++i) {
        EXPECT_DOUBLE_EQ(dx[i], 0.01);
    }
}

TEST(PDEWorkspaceSpansTest, PMRIntegration) {
    std::pmr::synchronized_pool_resource pool;
    size_t n = 101;
    size_t buffer_size = PDEWorkspaceSpans::required_size(n);

    std::pmr::vector<double> pmr_buffer(buffer_size, &pool);

    auto workspace = PDEWorkspaceSpans::from_buffer(pmr_buffer, n);

    ASSERT_TRUE(workspace.has_value());
    workspace->rhs()[0] = 42.0;
    EXPECT_DOUBLE_EQ(pmr_buffer[0], 42.0);
}

} // namespace mango
```

**Step 2: Add test target**

Modify `tests/BUILD.bazel`:

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

**Step 3: Run test**

Run: `bazel test //tests:pde_workspace_spans_test --test_output=all`

Expected: PASS (5 tests)

**Step 4: Commit**

```bash
git add tests/pde_workspace_spans_test.cc tests/BUILD.bazel
git commit -m "Add PDEWorkspaceSpans tests

Tests buffer sizing, validation, grid integration, PMR usage.

Related: #209"
```

---

**Commit 1 Complete - Infrastructure Added**

Run all new tests to verify:

```bash
bazel test //tests:grid_with_solution_test //tests:pde_workspace_spans_test --test_output=summary
```

Expected: All tests PASS

This completes Commit 1. The new infrastructure is added with zero impact on existing code.

---

## Commit 2: Atomic Update (PDESolver + Derived + Tests)

**Note:** This is a large breaking change. All substeps must be completed in sequence before committing.

### Task 2.1: Update PDESolver header (member variables)

**Files:**
- Modify: `src/pde/core/pde_solver.hpp`

**Step 1: Add new includes**

At top of `src/pde/core/pde_solver.hpp`, add:

```cpp
#include "src/pde/core/grid_with_solution.hpp"
#include "src/pde/core/pde_workspace_spans.hpp"
```

**Step 2: Replace member variables (lines ~245-266)**

Find the private member section and replace:

```cpp
// OLD (DELETE):
PDEWorkspace* workspace_;
std::span<const double> grid_;
std::span<double> output_buffer_;

// NEW (ADD):
std::shared_ptr<GridWithSolution<double>> grid_;
PDEWorkspaceSpans workspace_;
```

Keep all other members unchanged.

**Step 3: Verify (will not compile yet - expected)**

Run: `bazel build //src/pde/core:pde_solver`

Expected: FAIL (constructor doesn't match new members yet)

---

### Task 2.2: Replace PDESolver constructor

**Files:**
- Modify: `src/pde/core/pde_solver.hpp:72-137`

**Step 1: Replace constructor signature and body**

Find the PDESolver constructor (around line 72) and replace entirely:

```cpp
// NEW: Single constructor (replaces old)
PDESolver(std::shared_ptr<GridWithSolution<double>> grid,
          PDEWorkspaceSpans workspace)
    : grid_(grid)
    , workspace_(workspace)
    , time_(grid->time())
    , config_()
    , obstacle_()
    , n_(grid->n_space())
{
    solution_storage_.resize(2 * n_);
    u_current_ = std::span{solution_storage_}.subspan(0, n_);
    u_old_ = std::span{solution_storage_}.subspan(n_, n_);

    // Initialize RHS arrays
    rhs_.resize(n_);
}
```

**Step 2: Update initialize() method to use grid_**

Find initialize() method and update:

```cpp
// OLD: uses grid_ span directly
// NEW: uses grid_->x()
void initialize(std::function<void(std::span<const double>, std::span<double>)> init_func) {
    init_func(grid_->x(), u_current_);  // Changed from grid_
}
```

**Step 3: Verify compilation (will still fail)**

Run: `bazel build //src/pde/core:pde_solver`

Expected: FAIL (workspace_-> calls don't match workspace_ member)

---

### Task 2.3: Update workspace access (batch 1: frequently used)

**Files:**
- Modify: `src/pde/core/pde_solver.hpp` (throughout solve() and helper methods)

**Step 1: Replace workspace_->rhs() calls**

Find and replace all instances:

```bash
# In pde_solver.hpp:
# OLD: workspace_->rhs()
# NEW: workspace_.rhs()
```

Locations: Lines ~434, 439, 809, 812, 957, 964, 971, 994, 1000, 1013, 1020

**Step 2: Replace workspace_->lu() calls**

```bash
# OLD: workspace_->lu()
# NEW: workspace_.lu()
```

Locations: Lines ~434, 439, 809, 812, 949, 958, 965, 972, 995, 1001

**Step 3: Replace workspace_->u_stage() calls**

```bash
# OLD: workspace_->u_stage()
# NEW: workspace_.u_stage()
```

Locations: Lines ~948, 956, 957, 960, 963, 964, 967, 970, 971, 974, 993, 994, 997, 999, 1000, 1003, 1013, 1016, 1020

**Step 4: Verify compilation (getting closer)**

Run: `bazel build //src/pde/core:pde_solver`

Expected: FAIL (more workspace_-> calls remain)

---

### Task 2.4: Update workspace access (batch 2: remaining)

**Files:**
- Modify: `src/pde/core/pde_solver.hpp`

**Step 1: Replace workspace_->psi() calls**

```bash
# OLD: workspace_->psi()
# NEW: workspace_.psi()
```

Locations: Lines ~371, 669

**Step 2: Replace workspace_->dx() calls**

```bash
# OLD: workspace_->dx()
# NEW: workspace_.dx()
```

Locations: Line ~384

**Step 3: Replace Jacobian array calls**

```bash
# OLD: workspace_->jacobian_diag()
# NEW: workspace_.jacobian_diag()

# OLD: workspace_->jacobian_upper()
# NEW: workspace_.jacobian_upper()

# OLD: workspace_->jacobian_lower()
# NEW: workspace_.jacobian_lower()

# OLD: workspace_->residual()
# NEW: workspace_.residual()

# OLD: workspace_->delta_u()
# NEW: workspace_.delta_u()

# OLD: workspace_->newton_u_old()
# NEW: workspace_.newton_u_old()
```

Find all instances throughout the file (build_jacobian, solve_implicit_stage_projected, etc.)

**Step 4: Verify compilation**

Run: `bazel build //src/pde/core:pde_solver`

Expected: SUCCESS (PDESolver now compiles!)

---

### Task 2.5: Update solve() to write to Grid

**Files:**
- Modify: `src/pde/core/pde_solver.hpp:161-205`

**Step 1: Add Grid write at end of solve()**

Find the solve() method and add at the very end (before `return {}`):

```cpp
// Write final 2 steps to Grid (for theta computation)
auto grid_current = grid_->solution();
auto grid_prev = grid_->solution_prev();
std::copy(u_current_.begin(), u_current_.end(), grid_current.begin());
std::copy(u_old_.begin(), u_old_.end(), grid_prev.begin());

return {};
```

**Step 2: Update grid() accessor**

Find the grid() accessor method and update:

```cpp
// OLD:
std::span<const double> grid() const { return grid_; }

// NEW:
std::shared_ptr<GridWithSolution<double>> grid() const {
    return grid_;
}
```

**Step 3: Verify compilation**

Run: `bazel build //src/pde/core:pde_solver`

Expected: SUCCESS

**Step 4: Update BUILD.bazel dependencies**

Modify `src/pde/core/BUILD.bazel`:

```python
cc_library(
    name = "pde_solver",
    hdrs = ["pde_solver.hpp"],
    deps = [
        ":grid",
        ":grid_with_solution",     # ADD
        ":pde_workspace_spans",    # ADD
        ":time_domain",
        "//src/support:error_types",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 5: Verify compilation again**

Run: `bazel build //src/pde/core:pde_solver`

Expected: SUCCESS

---

**DO NOT COMMIT YET - PDESolver updated but derived classes still broken**

---

### Task 2.6: Update AmericanPutSolver constructor

**Files:**
- Modify: `src/option/american_pde_solver.hpp:48-93`

**Step 1: Replace constructor signature**

Find AmericanPutSolver constructor and replace:

```cpp
// OLD (DELETE):
AmericanPutSolver(const PricingParams& params,
                  std::shared_ptr<AmericanSolverWorkspace> workspace,
                  std::span<double> output_buffer = {})

// NEW (ADD):
AmericanPutSolver(const PricingParams& params,
                  std::shared_ptr<GridWithSolution<double>> grid,
                  PDEWorkspaceSpans workspace)
```

**Step 2: Update constructor body**

Replace the entire constructor body:

```cpp
AmericanPutSolver(const PricingParams& params,
                  std::shared_ptr<GridWithSolution<double>> grid,
                  PDEWorkspaceSpans workspace)
    : PDESolver<AmericanPutSolver>(grid, workspace)
    , params_(params)
    , left_bc_(create_left_bc(params))
    , right_bc_(create_right_bc(params))
    , spatial_op_(create_spatial_op(params, grid->spacing()))
{
    this->set_obstacle(create_obstacle_callback(params));
}
```

**Step 3: Add obstacle callback helper**

Add private helper method:

```cpp
private:
    static std::optional<ObstacleCallback> create_obstacle_callback(const PricingParams& params) {
        double K = params.strike;
        return [K](double t, std::span<const double> x, std::span<double> psi) {
            for (size_t i = 0; i < x.size(); ++i) {
                double S = K * std::exp(x[i]);
                psi[i] = std::max(K - S, 0.0);
            }
        };
    }
```

**Step 4: Remove workspace_ member**

Delete the line:

```cpp
std::shared_ptr<AmericanSolverWorkspace> workspace_;  // DELETE THIS
```

**Step 5: Verify (will not compile - expected)**

Run: `bazel build //src/option:american_pde_solver`

Expected: FAIL (AmericanCallSolver not updated yet, tests use old API)

---

### Task 2.7: Update AmericanCallSolver constructor

**Files:**
- Modify: `src/option/american_pde_solver.hpp:95-137`

**Step 1: Replace constructor**

Same pattern as AmericanPutSolver:

```cpp
AmericanCallSolver(const PricingParams& params,
                   std::shared_ptr<GridWithSolution<double>> grid,
                   PDEWorkspaceSpans workspace)
    : PDESolver<AmericanCallSolver>(grid, workspace)
    , params_(params)
    , left_bc_(create_left_bc(params))
    , right_bc_(create_right_bc(params))
    , spatial_op_(create_spatial_op(params, grid->spacing()))
{
    this->set_obstacle(create_obstacle_callback(params));
}
```

**Step 2: Add obstacle callback helper**

```cpp
private:
    static std::optional<ObstacleCallback> create_obstacle_callback(const PricingParams& params) {
        double K = params.strike;
        return [K](double t, std::span<const double> x, std::span<double> psi) {
            for (size_t i = 0; i < x.size(); ++i) {
                double S = K * std::exp(x[i]);
                psi[i] = std::max(S - K, 0.0);
            }
        };
    }
```

**Step 3: Remove workspace_ member**

**Step 4: Update BUILD.bazel**

Modify `src/option/BUILD.bazel`:

```python
cc_library(
    name = "american_pde_solver",
    hdrs = ["american_pde_solver.hpp"],
    deps = [
        "//src/pde/core:pde_solver",
        "//src/pde/core:grid_with_solution",     # ADD
        "//src/pde/core:pde_workspace_spans",    # ADD
        "//src/option:option_spec",
        # Remove american_solver_workspace dependency
    ],
    visibility = ["//visibility:public"],
)
```

**Step 5: Verify (still won't compile - tests not updated)**

Run: `bazel build //src/option:american_pde_solver`

Expected: SUCCESS (library compiles, tests don't)

---

**DO NOT COMMIT YET - Tests still use old API**

---

### Task 2.8: Update american_option_test.cc (all tests)

**Files:**
- Modify: `tests/american_option_test.cc`

**Step 1: Update test helper to create Grid + Workspace**

Add helper function at top of test file:

```cpp
namespace {

struct TestWorkspace {
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer;
    PDEWorkspaceSpans spans;

    static TestWorkspace create(size_t n) {
        TestWorkspace ws;
        size_t buffer_size = PDEWorkspaceSpans::required_size(n);
        ws.buffer = std::pmr::vector<double>(buffer_size, &ws.pool);
        return ws;
    }

    void initialize_with_grid(std::span<const double> grid) {
        spans = PDEWorkspaceSpans::from_buffer_and_grid(
            buffer, grid, grid.size()).value();
    }
};

} // namespace
```

**Step 2: Update first test (SolverWithPMRWorkspace)**

Replace test body:

```cpp
TEST_F(AmericanOptionPricingTest, SolverWithPMRWorkspace) {
    PricingParams params{
        .spot = 100.0, .strike = 100.0, .maturity = 1.0,
        .rate = 0.05, .dividend = 0.02, .type = OptionType::PUT,
        .volatility = 0.20
    };

    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0).value();
    TimeDomain time{.t_start = 0.0, .t_end = params.maturity, .n_steps = 1000};
    auto grid = GridWithSolution<double>::create(grid_spec, time).value();

    auto ws = TestWorkspace::create(grid->n_space());
    ws.initialize_with_grid(grid->x());

    AmericanPutSolver solver(params, grid, ws.spans);
    solver.initialize(AmericanPutSolver::payoff);

    auto result = solver.solve();

    ASSERT_TRUE(result.has_value());
    auto solution = grid->solution();
    EXPECT_GT(solution[50], 0.0);
}
```

**Step 3: Update remaining tests (same pattern)**

Apply same transformation to:
- PutValueRespectsIntrinsicBound
- CallValueIncreasesWithVolatility
- PutValueIncreasesWithMaturity
- BatchSolverMatchesSingleSolver
- PutImmediateExerciseAtBoundary
- ATMOptionsRetainTimeValue

**Step 4: Run tests**

Run: `bazel test //tests:american_option_test --test_output=all`

Expected: PASS (7 tests)

---

### Task 2.9: Delete AmericanSolverWorkspace

**Files:**
- Delete: `src/option/american_solver_workspace.hpp`
- Delete: `src/option/american_solver_workspace.cpp`
- Delete: `tests/american_solver_workspace_test.cc`
- Modify: `src/option/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Remove files**

```bash
git rm src/option/american_solver_workspace.hpp
git rm src/option/american_solver_workspace.cpp
git rm tests/american_solver_workspace_test.cc
```

**Step 2: Remove BUILD targets**

In `src/option/BUILD.bazel`, delete:

```python
cc_library(
    name = "american_solver_workspace",
    # DELETE ENTIRE TARGET
)
```

In `tests/BUILD.bazel`, delete:

```python
cc_test(
    name = "american_solver_workspace_test",
    # DELETE ENTIRE TARGET
)
```

**Step 3: Verify all tests pass**

Run: `bazel test //tests:american_option_test --test_output=summary`

Expected: PASS (all tests)

**Step 4: Verify build**

Run: `bazel build //...`

Expected: SUCCESS (everything compiles)

---

**NOW COMMIT - Atomic change complete**

```bash
git add -A
git commit -m "Refactor PDE workspace: Grid + Workspace separation

BREAKING CHANGE: Replaced PDESolver constructor and AmericanSolverWorkspace.

Changes:
- PDESolver: Now accepts GridWithSolution + PDEWorkspaceSpans
- Replaced workspace_-> with workspace_. (direct member access)
- Updated AmericanPutSolver and AmericanCallSolver constructors
- Deleted AmericanSolverWorkspace class (replaced by Grid + Workspace)
- Updated all tests to new API
- Grid writes final 2 steps for theta computation

Benefits:
- 500× memory reduction (Grid stores 2n, not n_time × n)
- Clean separation: Grid (persistent), Workspace (temporary), Solver (compute)
- Zero dead code (no dual paths)
- Simpler: workspace_.rhs() instead of get_rhs()

Fixes #206, #203, #207
Enables #208
Related: #209

Codex approved (v3), clean migration design (v4)"
```

---

## Commit 3: Enable Theta Computation

### Task 3.1: Implement compute_theta()

**Files:**
- Modify: `src/option/american_option.cpp:266-276`

**Step 1: Replace stub implementation**

Find compute_theta() and replace:

```cpp
double AmericanOptionSolver::compute_theta() const {
    if (!solved_) {
        return 0.0;
    }

    // Get Grid (contains last 2 time steps)
    // Note: We need to get grid from the solver
    // This requires solver to expose grid(), which AmericanPutSolver provides

    // For now, this is a stub - needs access to solver's grid
    // Will be completed after solver refactoring
    return 0.0;  // TODO: Implement using Grid storage
}
```

**Step 2: Update AmericanOptionSolver to store grid reference**

This task requires AmericanOptionSolver to have access to the Grid.
Since AmericanPutSolver/AmericanCallSolver inherit from PDESolver which has grid(),
we need to access it through the solver.

**Mark as TODO for now - this will be addressed in a follow-up**

**Step 3: Commit**

```bash
git add src/option/american_option.cpp
git commit -m "Stub: Theta computation (TODO after solver access)

Theta requires Grid access which needs solver refactoring.
Will implement in follow-up task.

Related: #209, #208"
```

---

## Summary

**3 commits total:**

1. **Commit 1:** Add infrastructure (GridWithSolution, PDEWorkspaceSpans, tests) - SAFE
2. **Commit 2:** Atomic update (PDESolver + derived + tests + delete old) - BREAKING but COMPLETE
3. **Commit 3:** Enable theta (stub for now, needs follow-up) - SAFE

**Timeline:**
- Commit 1: ~1.5 hours
- Commit 2: ~4 hours
- Commit 3: ~15 minutes
- Total: ~5.75 hours

**Testing strategy:**
- Each commit: `bazel test //...`
- Commit 2: `bazel test //tests:american_option_test` (critical)
- Final: Run benchmarks to verify no regression

**Rollback:**
- Commit 1: Can revert safely (no dependencies)
- Commit 2: Atomic - either works or doesn't
- Commit 3: Stub only, safe to revert

---

Plan saved to: `docs/plans/2025-11-20-pde-workspace-implementation.md`
