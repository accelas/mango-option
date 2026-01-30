# PMR-Based Memory Management Refactoring

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor PDE solver memory management to use PMR (Polymorphic Memory Resources) throughout, separating compute from allocation with thread-safe resource pools.

**Architecture:** Replace raw pointer-based PDEWorkspace with pmr::vector storage, eliminate GridBuffer in favor of direct spans, merge Newton arrays into unified workspace, use synchronized_pool_resource for workspace and default resource for persistent results.

**Tech Stack:** C++23, std::pmr, std::expected, Bazel

---

## Phase 1: PDEWorkspace PMR Refactor

### Task 1.1: Create New PDEWorkspace with PMR Vectors

**Files:**
- Create: `src/pde/core/pde_workspace_pmr.hpp`
- Test: `tests/pde_workspace_pmr_test.cc`

**Step 1: Write the failing test**

Create `tests/pde_workspace_pmr_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "src/pde/core/pde_workspace_pmr.hpp"
#include <memory_resource>

namespace mango {
namespace {

TEST(PDEWorkspacePMRTest, FactoryCreatesWorkspace) {
    std::pmr::synchronized_pool_resource pool;

    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_spec.has_value());

    auto workspace = PDEWorkspace::create(grid_spec.value(), &pool);
    ASSERT_TRUE(workspace.has_value());

    auto ws = workspace.value();
    EXPECT_EQ(ws->logical_size(), 101);
    EXPECT_EQ(ws->padded_size(), 104);  // Rounded to SIMD_WIDTH=8
}

TEST(PDEWorkspacePMRTest, AccessorsReturnPaddedSpans) {
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    auto ws = PDEWorkspace::create(grid_spec.value(), &pool).value();

    auto u_current = ws->u_current();
    EXPECT_EQ(u_current.size(), 104);  // Padded size

    // Check we can write to all elements
    for (size_t i = 0; i < u_current.size(); ++i) {
        u_current[i] = static_cast<double>(i);
    }
}

TEST(PDEWorkspacePMRTest, GridAccessReturnsCorrectData) {
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    auto ws = PDEWorkspace::create(grid_spec.value(), &pool).value();

    auto grid = ws->grid();
    EXPECT_EQ(grid.size(), 104);  // Padded
    EXPECT_NEAR(grid[0], 0.0, 1e-14);
    EXPECT_NEAR(grid[100], 1.0, 1e-14);
}

}  // namespace
}  // namespace mango
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:pde_workspace_pmr_test --test_output=errors`
Expected: FAIL with "No such file or directory: src/pde/core/pde_workspace_pmr.hpp"

**Step 3: Write minimal implementation**

Create `src/pde/core/pde_workspace_pmr.hpp`:

```cpp
#pragma once

#include "src/pde/core/grid.hpp"
#include <memory_resource>
#include <span>
#include <expected>
#include <string>
#include <memory>
#include <algorithm>

namespace mango {

/**
 * PDEWorkspace: Unified memory workspace for PDE solver
 *
 * Uses PMR vectors for all storage. All accessors return SIMD-padded spans.
 * Caller extracts logical size with .subspan(0, logical_size()) when needed.
 */
class PDEWorkspace {
public:
    static constexpr size_t SIMD_WIDTH = 8;

    static constexpr size_t pad_to_simd(size_t n) {
        return ((n + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
    }

    static std::expected<std::shared_ptr<PDEWorkspace>, std::string>
    create(const GridSpec<double>& grid_spec,
           std::pmr::memory_resource* resource) {
        if (!resource) {
            return std::unexpected("Memory resource cannot be null");
        }

        auto grid_result = grid_spec.generate();
        if (!grid_result.has_value()) {
            return std::unexpected(grid_result.error());
        }

        auto grid_buffer = grid_result.value();
        size_t n = grid_buffer.size();

        if (n == 0) {
            return std::unexpected("Grid size must be positive");
        }

        return std::shared_ptr<PDEWorkspace>(
            new PDEWorkspace(n, grid_buffer.span(), resource));
    }

    // Accessors - all return SIMD-padded spans
    std::span<double> u_current() { return {u_current_.data(), padded_n_}; }
    std::span<const double> u_current() const { return {u_current_.data(), padded_n_}; }

    std::span<double> u_next() { return {u_next_.data(), padded_n_}; }
    std::span<const double> u_next() const { return {u_next_.data(), padded_n_}; }

    std::span<double> u_stage() { return {u_stage_.data(), padded_n_}; }
    std::span<const double> u_stage() const { return {u_stage_.data(), padded_n_}; }

    std::span<double> rhs() { return {rhs_.data(), padded_n_}; }
    std::span<const double> rhs() const { return {rhs_.data(), padded_n_}; }

    std::span<double> lu() { return {lu_.data(), padded_n_}; }
    std::span<const double> lu() const { return {lu_.data(), padded_n_}; }

    std::span<double> psi() { return {psi_.data(), padded_n_}; }
    std::span<const double> psi() const { return {psi_.data(), padded_n_}; }

    std::span<const double> grid() const { return {grid_.data(), padded_n_}; }

    std::span<const double> dx() const { return {dx_.data(), pad_to_simd(n_ - 1)}; }

    size_t logical_size() const { return n_; }
    size_t padded_size() const { return padded_n_; }

private:
    PDEWorkspace(size_t n, std::span<const double> grid_data,
                 std::pmr::memory_resource* mr)
        : n_(n)
        , padded_n_(pad_to_simd(n))
        , resource_(mr)
        , grid_(padded_n_, 0.0, mr)
        , u_current_(padded_n_, 0.0, mr)
        , u_next_(padded_n_, 0.0, mr)
        , u_stage_(padded_n_, 0.0, mr)
        , rhs_(padded_n_, 0.0, mr)
        , lu_(padded_n_, 0.0, mr)
        , psi_(padded_n_, 0.0, mr)
        , dx_(pad_to_simd(n - 1), 0.0, mr)
    {
        // Copy grid data
        std::copy(grid_data.begin(), grid_data.end(), grid_.begin());

        // Precompute dx
        for (size_t i = 0; i < n_ - 1; ++i) {
            dx_[i] = grid_[i + 1] - grid_[i];
        }
    }

    size_t n_;
    size_t padded_n_;
    std::pmr::memory_resource* resource_;

    std::pmr::vector<double> grid_;
    std::pmr::vector<double> u_current_;
    std::pmr::vector<double> u_next_;
    std::pmr::vector<double> u_stage_;
    std::pmr::vector<double> rhs_;
    std::pmr::vector<double> lu_;
    std::pmr::vector<double> psi_;
    std::pmr::vector<double> dx_;
};

}  // namespace mango
```

**Step 4: Add BUILD rule**

Modify `src/pde/core/BUILD.bazel`, add:

```python
cc_library(
    name = "pde_workspace_pmr",
    hdrs = ["pde_workspace_pmr.hpp"],
    deps = [
        ":grid",
    ],
    visibility = ["//visibility:public"],
)
```

Modify `tests/BUILD.bazel`, add:

```python
cc_test(
    name = "pde_workspace_pmr_test",
    srcs = ["pde_workspace_pmr_test.cc"],
    deps = [
        "//src/pde/core:pde_workspace_pmr",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Run test to verify it passes**

Run: `bazel test //tests:pde_workspace_pmr_test --test_output=errors`
Expected: PASS

**Step 6: Commit**

```bash
git add src/pde/core/pde_workspace_pmr.hpp \
        src/pde/core/BUILD.bazel \
        tests/pde_workspace_pmr_test.cc \
        tests/BUILD.bazel
git commit -m "Add PDEWorkspace with PMR vector storage

- Factory method creates workspace from GridSpec
- All arrays stored in pmr::vector with SIMD padding
- Accessors return padded spans
- Precomputed dx array for grid spacing"
```

---

### Task 1.2: Add Newton Arrays to PDEWorkspace

**Files:**
- Modify: `src/pde/core/pde_workspace_pmr.hpp`
- Modify: `tests/pde_workspace_pmr_test.cc`

**Step 1: Write the failing test**

Add to `tests/pde_workspace_pmr_test.cc`:

```cpp
TEST(PDEWorkspacePMRTest, NewtonArraysAccessible) {
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    auto ws = PDEWorkspace::create(grid_spec.value(), &pool).value();

    // Test Newton array access
    auto jac_diag = ws->jacobian_diag();
    auto jac_upper = ws->jacobian_upper();
    auto jac_lower = ws->jacobian_lower();
    auto residual = ws->residual();
    auto delta_u = ws->delta_u();

    EXPECT_EQ(jac_diag.size(), 104);
    EXPECT_EQ(jac_upper.size(), 104);
    EXPECT_EQ(jac_lower.size(), 104);
    EXPECT_EQ(residual.size(), 104);
    EXPECT_EQ(delta_u.size(), 104);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:pde_workspace_pmr_test --test_output=errors`
Expected: FAIL with "no member named 'jacobian_diag' in 'mango::PDEWorkspace'"

**Step 3: Add Newton array members**

Modify `src/pde/core/pde_workspace_pmr.hpp`, add to public accessors:

```cpp
    // Newton solver arrays
    std::span<double> jacobian_diag() { return {jacobian_diag_.data(), padded_n_}; }
    std::span<const double> jacobian_diag() const { return {jacobian_diag_.data(), padded_n_}; }

    std::span<double> jacobian_upper() { return {jacobian_upper_.data(), padded_n_}; }
    std::span<const double> jacobian_upper() const { return {jacobian_upper_.data(), padded_n_}; }

    std::span<double> jacobian_lower() { return {jacobian_lower_.data(), padded_n_}; }
    std::span<const double> jacobian_lower() const { return {jacobian_lower_.data(), padded_n_}; }

    std::span<double> residual() { return {residual_.data(), padded_n_}; }
    std::span<const double> residual() const { return {residual_.data(), padded_n_}; }

    std::span<double> delta_u() { return {delta_u_.data(), padded_n_}; }
    std::span<const double> delta_u() const { return {delta_u_.data(), padded_n_}; }
```

Add to private members:

```cpp
    std::pmr::vector<double> jacobian_diag_;
    std::pmr::vector<double> jacobian_upper_;
    std::pmr::vector<double> jacobian_lower_;
    std::pmr::vector<double> residual_;
    std::pmr::vector<double> delta_u_;
```

Add to constructor initializer list:

```cpp
        , jacobian_diag_(padded_n_, 0.0, mr)
        , jacobian_upper_(padded_n_, 0.0, mr)
        , jacobian_lower_(padded_n_, 0.0, mr)
        , residual_(padded_n_, 0.0, mr)
        , delta_u_(padded_n_, 0.0, mr)
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:pde_workspace_pmr_test --test_output=errors`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pde/core/pde_workspace_pmr.hpp tests/pde_workspace_pmr_test.cc
git commit -m "Add Newton solver arrays to PDEWorkspace

Merge Newton arrays into unified workspace:
- jacobian_diag, jacobian_upper, jacobian_lower
- residual, delta_u
All allocated from same PMR resource"
```

---

## Phase 2: GridSpacing as Value Type

### Task 2.1: Refactor GridSpacing to Use Span Views

**Files:**
- Modify: `src/pde/core/grid_spacing.hpp`
- Create: `tests/grid_spacing_view_test.cc`

**Step 1: Write the failing test**

Create `tests/grid_spacing_view_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "src/pde/core/grid_spacing.hpp"
#include <vector>

namespace mango {
namespace {

TEST(GridSpacingViewTest, CreateFromSpans) {
    std::vector<double> grid = {0.0, 0.1, 0.2, 0.3};
    std::vector<double> dx = {0.1, 0.1, 0.1};

    auto spacing = GridSpacing::create(grid, dx);
    ASSERT_TRUE(spacing.has_value());

    EXPECT_TRUE(spacing->is_uniform());
    EXPECT_NEAR(spacing->spacing(), 0.1, 1e-14);
}

TEST(GridSpacingViewTest, ValueTypeSemantics) {
    std::vector<double> grid = {0.0, 0.1, 0.2};
    std::vector<double> dx = {0.1, 0.1};

    auto spacing1 = GridSpacing::create(grid, dx).value();
    auto spacing2 = spacing1;  // Copy

    EXPECT_TRUE(spacing2.is_uniform());
    EXPECT_NEAR(spacing2.spacing(), 0.1, 1e-14);
}

TEST(GridSpacingViewTest, NonUniformSpacing) {
    std::vector<double> grid = {0.0, 0.05, 0.15, 0.3};
    std::vector<double> dx = {0.05, 0.10, 0.15};

    auto spacing = GridSpacing::create(grid, dx);
    ASSERT_TRUE(spacing.has_value());

    EXPECT_FALSE(spacing->is_uniform());
    auto dx_left = spacing->dx_left_inv();
    EXPECT_GT(dx_left.size(), 0);
}

}  // namespace
}  // namespace mango
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:grid_spacing_view_test --test_output=errors`
Expected: FAIL (compilation errors or wrong behavior)

**Step 3: Refactor GridSpacing**

Modify `src/pde/core/grid_spacing.hpp`:

```cpp
#pragma once

#include <span>
#include <variant>
#include <expected>
#include <string>
#include <cmath>
#include <vector>

namespace mango {

struct UniformSpacing {
    double dx;
    double dx_inv;
    double dx_inv_sq;
};

struct NonUniformSpacing {
    std::span<const double> grid;
    std::span<const double> dx;
    // Precomputed arrays stored in workspace, viewed here
    std::vector<double> dx_left_inv_data;
    std::vector<double> dx_right_inv_data;
    std::vector<double> dx_center_inv_data;
    std::vector<double> w_left_data;
    std::vector<double> w_right_data;
};

class GridSpacing {
public:
    static std::expected<GridSpacing, std::string>
    create(std::span<const double> grid, std::span<const double> dx) {
        if (grid.size() < 2) {
            return std::unexpected("Grid must have at least 2 points");
        }
        if (dx.size() != grid.size() - 1) {
            return std::unexpected("dx size must be grid.size() - 1");
        }

        // Check if uniform
        const double dx0 = dx[0];
        constexpr double tol = 1e-10;
        bool uniform = true;
        for (size_t i = 1; i < dx.size(); ++i) {
            if (std::abs(dx[i] - dx0) > tol) {
                uniform = false;
                break;
            }
        }

        if (uniform) {
            return GridSpacing(UniformSpacing{
                dx0, 1.0 / dx0, 1.0 / (dx0 * dx0)
            });
        } else {
            return GridSpacing(grid, dx);
        }
    }

    bool is_uniform() const {
        return std::holds_alternative<UniformSpacing>(spacing_);
    }

    double spacing() const {
        return std::get<UniformSpacing>(spacing_).dx;
    }

    double spacing_inv() const {
        return std::get<UniformSpacing>(spacing_).dx_inv;
    }

    std::span<const double> dx_left_inv() const {
        const auto& nu = std::get<NonUniformSpacing>(spacing_);
        return nu.dx_left_inv_data;
    }

    std::span<const double> dx_right_inv() const {
        const auto& nu = std::get<NonUniformSpacing>(spacing_);
        return nu.dx_right_inv_data;
    }

    std::span<const double> dx_center_inv() const {
        const auto& nu = std::get<NonUniformSpacing>(spacing_);
        return nu.dx_center_inv_data;
    }

    std::span<const double> w_left() const {
        const auto& nu = std::get<NonUniformSpacing>(spacing_);
        return nu.w_left_data;
    }

    std::span<const double> w_right() const {
        const auto& nu = std::get<NonUniformSpacing>(spacing_);
        return nu.w_right_data;
    }

private:
    explicit GridSpacing(UniformSpacing us)
        : spacing_(us)
    {}

    GridSpacing(std::span<const double> grid, std::span<const double> dx)
        : spacing_(create_nonuniform(grid, dx))
    {}

    static NonUniformSpacing create_nonuniform(
        std::span<const double> grid,
        std::span<const double> dx)
    {
        size_t n = grid.size();
        NonUniformSpacing nu;
        nu.grid = grid;
        nu.dx = dx;

        // Allocate precomputed arrays
        nu.dx_left_inv_data.resize(n - 2);
        nu.dx_right_inv_data.resize(n - 2);
        nu.dx_center_inv_data.resize(n - 2);
        nu.w_left_data.resize(n - 2);
        nu.w_right_data.resize(n - 2);

        // Precompute values
        for (size_t i = 0; i < n - 2; ++i) {
            double dx_left = dx[i];
            double dx_right = dx[i + 1];
            double dx_center = dx_left + dx_right;

            nu.dx_left_inv_data[i] = 1.0 / dx_left;
            nu.dx_right_inv_data[i] = 1.0 / dx_right;
            nu.dx_center_inv_data[i] = 1.0 / dx_center;
            nu.w_left_data[i] = dx_right / dx_center;
            nu.w_right_data[i] = dx_left / dx_center;
        }

        return nu;
    }

    std::variant<UniformSpacing, NonUniformSpacing> spacing_;
};

}  // namespace mango
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:grid_spacing_view_test --test_output=errors`
Expected: PASS

**Step 5: Update BUILD file**

Modify `tests/BUILD.bazel`, add:

```python
cc_test(
    name = "grid_spacing_view_test",
    srcs = ["grid_spacing_view_test.cc"],
    deps = [
        "//src/pde/core:grid_spacing",
        "@googletest//:gtest_main",
    ],
)
```

**Step 6: Commit**

```bash
git add src/pde/core/grid_spacing.hpp \
        tests/grid_spacing_view_test.cc \
        tests/BUILD.bazel
git commit -m "Refactor GridSpacing to value type with span views

- Accept grid and dx as spans (no ownership)
- Value type semantics (copy/move cheap for uniform)
- NonUniform variant still allocates precomputed arrays
- Factory method validates input"
```

---

## Phase 3: Update PDESolver for PMR Workspace

### Task 3.1: Simplify PDESolver Constructor

**Files:**
- Modify: `src/pde/core/pde_solver.hpp`
- Modify: `tests/pde_solver_test.cc`

**Step 1: Write the failing test**

Add to `tests/pde_solver_test.cc`:

```cpp
TEST(PDESolverTest, ConstructWithWorkspaceOnly) {
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    auto workspace = PDEWorkspace::create(grid_spec.value(), &pool).value();

    TimeDomain time(0.0, 0.1, 0.001);

    DirichletBC left_bc{[](double, double) { return 0.0; }};
    DirichletBC right_bc{[](double, double) { return 0.0; }};

    auto pde = operators::LaplacianPDE<double>(1.0);
    auto grid_view = GridView<double>(workspace->grid().subspan(0, 101));
    auto spatial_op = operators::create_spatial_operator(std::move(pde), grid_view);

    auto solver = make_test_solver(workspace->grid().subspan(0, 101),
                                   time, left_bc, right_bc, spatial_op);

    // Should have default TRBDF2Config
    EXPECT_EQ(solver.config().max_iter, 20);
    EXPECT_NEAR(solver.config().tolerance, 1e-6, 1e-10);
}

TEST(PDESolverTest, SetConfigChangesSettings) {
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);
    auto workspace = PDEWorkspace::create(grid_spec.value(), &pool).value();

    TimeDomain time(0.0, 0.1, 0.001);

    DirichletBC left_bc{[](double, double) { return 0.0; }};
    DirichletBC right_bc{[](double, double) { return 0.0; }};

    auto pde = operators::LaplacianPDE<double>(1.0);
    auto grid_view = GridView<double>(workspace->grid().subspan(0, 101));
    auto spatial_op = operators::create_spatial_operator(std::move(pde), grid_view);

    auto solver = make_test_solver(workspace->grid().subspan(0, 101),
                                   time, left_bc, right_bc, spatial_op);

    TRBDF2Config new_config{.max_iter = 50, .tolerance = 1e-8};
    solver.set_config(new_config);

    EXPECT_EQ(solver.config().max_iter, 50);
    EXPECT_NEAR(solver.config().tolerance, 1e-8, 1e-10);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:pde_solver_test --test_output=errors`
Expected: FAIL with "no member named 'config' in PDESolver"

**Step 3: Modify PDESolver**

In `src/pde/core/pde_solver.hpp`, update constructor to remove config parameter:

```cpp
template<typename Derived>
class PDESolver {
public:
    PDESolver(std::span<const double> grid,
              const TimeDomain& time,
              std::optional<ObstacleCallback> obstacle = std::nullopt,
              PDEWorkspace* workspace = nullptr,
              std::span<double> output_buffer = {})
        : grid_(grid)
        , time_(time)
        , config_{}  // Default-initialized
        , obstacle_(obstacle)
        , workspace_(workspace)
        , output_buffer_(output_buffer)
    {}

    // Add setter and getter
    void set_config(const TRBDF2Config& config) {
        config_ = config;
    }

    const TRBDF2Config& config() const {
        return config_;
    }

    // ... rest of class unchanged

private:
    std::span<const double> grid_;
    TimeDomain time_;
    TRBDF2Config config_;  // Now a member, not constructor param
    std::optional<ObstacleCallback> obstacle_;
    PDEWorkspace* workspace_;
    std::span<double> output_buffer_;
};
```

**Step 4: Update TestPDESolver helper**

In `tests/pde_solver_test.cc`, update constructor:

```cpp
template<typename LeftBC, typename RightBC, typename SpatialOp>
class TestPDESolver : public mango::PDESolver<TestPDESolver<LeftBC, RightBC, SpatialOp>> {
public:
    TestPDESolver(std::span<const double> grid,
                  const mango::TimeDomain& time,
                  LeftBC left_bc,
                  RightBC right_bc,
                  SpatialOp spatial_op)
        : mango::PDESolver<TestPDESolver>(
              grid, time, std::nullopt, nullptr, {})
        , left_bc_(std::move(left_bc))
        , right_bc_(std::move(right_bc))
        , spatial_op_(std::move(spatial_op))
    {}

    // CRTP interface
    const LeftBC& left_boundary() const { return left_bc_; }
    const RightBC& right_boundary() const { return right_bc_; }
    const SpatialOp& spatial_operator() const { return spatial_op_; }

private:
    LeftBC left_bc_;
    RightBC right_bc_;
    SpatialOp spatial_op_;
};
```

Update `make_test_solver` to remove config param:

```cpp
template<typename LeftBC, typename RightBC, typename SpatialOp>
auto make_test_solver(std::span<const double> grid,
                      const mango::TimeDomain& time,
                      LeftBC left_bc,
                      RightBC right_bc,
                      SpatialOp spatial_op) {
    return TestPDESolver<LeftBC, RightBC, SpatialOp>(
        grid, time, std::move(left_bc), std::move(right_bc), std::move(spatial_op));
}
```

**Step 5: Run test to verify it passes**

Run: `bazel test //tests:pde_solver_test --test_output=errors`
Expected: PASS

**Step 6: Commit**

```bash
git add src/pde/core/pde_solver.hpp tests/pde_solver_test.cc
git commit -m "Simplify PDESolver constructor with default config

- Remove TRBDF2Config from constructor parameters
- Add config_ member with default initialization
- Add set_config() and config() methods
- Update tests to use new API"
```

---

## Phase 4: AmericanSolverWorkspace Integration

### Task 4.1: Update AmericanSolverWorkspace to Use PMR Workspace

**Files:**
- Modify: `src/option/american_solver_workspace.hpp`
- Modify: `src/option/american_solver_workspace.cpp`
- Modify: `tests/american_solver_workspace_test.cc`

**Step 1: Write the failing test**

Modify `tests/american_solver_workspace_test.cc`:

```cpp
TEST(AmericanSolverWorkspaceTest, CreateWithGridSpec) {
    std::pmr::synchronized_pool_resource pool;

    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0);

    auto workspace = AmericanSolverWorkspace::create(
        grid_spec.value(), 1000, &pool);

    ASSERT_TRUE(workspace.has_value());

    auto ws = workspace.value();
    EXPECT_EQ(ws->n_space(), 201);
    EXPECT_EQ(ws->n_time(), 1000);

    // Check we can access PDEWorkspace
    auto pde_ws = ws->pde_workspace();
    ASSERT_NE(pde_ws, nullptr);
    EXPECT_EQ(pde_ws->logical_size(), 201);
}

TEST(AmericanSolverWorkspaceTest, GridSpacingAvailable) {
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 101);

    auto workspace = AmericanSolverWorkspace::create(
        grid_spec.value(), 1000, &pool).value();

    auto spacing = workspace->grid_spacing();
    EXPECT_TRUE(spacing.is_uniform());
    EXPECT_NEAR(spacing.spacing(), 0.01, 1e-10);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:american_solver_workspace_test --test_output=errors`
Expected: FAIL (API mismatch)

**Step 3: Modify AmericanSolverWorkspace header**

In `src/option/american_solver_workspace.hpp`:

```cpp
#pragma once

#include "src/pde/core/pde_workspace_pmr.hpp"
#include "src/pde/core/grid_spacing.hpp"
#include "src/pde/core/grid.hpp"
#include <memory>
#include <expected>
#include <string>
#include <memory_resource>

namespace mango {

class AmericanSolverWorkspace {
public:
    static std::expected<std::shared_ptr<AmericanSolverWorkspace>, std::string>
    create(const GridSpec<double>& grid_spec,
           size_t n_time,
           std::pmr::memory_resource* resource);

    std::shared_ptr<PDEWorkspace> pde_workspace() const { return pde_workspace_; }
    GridSpacing grid_spacing() const { return grid_spacing_; }
    std::span<const double> grid() const { return pde_workspace_->grid(); }

    size_t n_space() const { return pde_workspace_->logical_size(); }
    size_t n_time() const { return n_time_; }

private:
    AmericanSolverWorkspace(std::shared_ptr<PDEWorkspace> pde_ws,
                           GridSpacing spacing,
                           size_t n_time)
        : pde_workspace_(std::move(pde_ws))
        , grid_spacing_(spacing)
        , n_time_(n_time)
    {}

    std::shared_ptr<PDEWorkspace> pde_workspace_;
    GridSpacing grid_spacing_;
    size_t n_time_;
};

}  // namespace mango
```

**Step 4: Implement factory method**

In `src/option/american_solver_workspace.cpp`:

```cpp
#include "src/option/american_solver_workspace.hpp"

namespace mango {

std::expected<std::shared_ptr<AmericanSolverWorkspace>, std::string>
AmericanSolverWorkspace::create(const GridSpec<double>& grid_spec,
                                size_t n_time,
                                std::pmr::memory_resource* resource) {
    if (!resource) {
        return std::unexpected("Memory resource cannot be null");
    }

    if (n_time == 0) {
        return std::unexpected("n_time must be positive");
    }

    // Create PDEWorkspace
    auto pde_ws_result = PDEWorkspace::create(grid_spec, resource);
    if (!pde_ws_result.has_value()) {
        return std::unexpected(pde_ws_result.error());
    }

    auto pde_ws = pde_ws_result.value();

    // Create GridSpacing view
    auto spacing_result = GridSpacing::create(
        pde_ws->grid().subspan(0, pde_ws->logical_size()),
        pde_ws->dx().subspan(0, pde_ws->logical_size() - 1));

    if (!spacing_result.has_value()) {
        return std::unexpected(spacing_result.error());
    }

    return std::shared_ptr<AmericanSolverWorkspace>(
        new AmericanSolverWorkspace(pde_ws, spacing_result.value(), n_time));
}

}  // namespace mango
```

**Step 5: Run test to verify it passes**

Run: `bazel test //tests:american_solver_workspace_test --test_output=errors`
Expected: PASS

**Step 6: Commit**

```bash
git add src/option/american_solver_workspace.hpp \
        src/option/american_solver_workspace.cpp \
        tests/american_solver_workspace_test.cc
git commit -m "Update AmericanSolverWorkspace to use PMR workspace

- Accept GridSpec and memory_resource in create()
- Create PDEWorkspace with PMR allocation
- Create GridSpacing as value type view
- Propagate resource through constructor chain"
```

---

## Phase 5: Update American Option Solvers

### Task 5.1: Update AmericanPutSolver for New Workspace API

**Files:**
- Modify: `src/option/american_pde_solver.hpp`
- Modify: `tests/american_option_test.cc`

**Step 1: Write the failing test**

Add to `tests/american_option_test.cc`:

```cpp
TEST_F(AmericanOptionPricingTest, SolverWithPMRWorkspace) {
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0);

    auto workspace = AmericanSolverWorkspace::create(
        grid_spec.value(), 2000, &pool);
    ASSERT_TRUE(workspace.has_value());

    AmericanOptionParams params(
        100.0,  // spot
        110.0,  // strike
        1.0,    // maturity
        0.03,   // rate
        0.00,   // dividend_yield
        OptionType::PUT,
        0.25    // volatility
    );

    auto solver_result = AmericanOptionSolver::create(params, workspace.value());
    ASSERT_TRUE(solver_result.has_value());

    auto result = solver_result.value().solve();
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->converged);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:american_option_test --test_output=errors`
Expected: FAIL (API changes needed)

**Step 3: Update AmericanPutSolver**

In `src/option/american_pde_solver.hpp`:

```cpp
class AmericanPutSolver : public PDESolver<AmericanPutSolver> {
public:
    AmericanPutSolver(const PricingParams& params,
                     std::shared_ptr<AmericanSolverWorkspace> workspace,
                     std::span<double> output_buffer = {})
        : PDESolver<AmericanPutSolver>(
              workspace->grid().subspan(0, workspace->n_space()),
              TimeDomain(0.0, params.maturity, params.maturity / workspace->n_time()),
              create_obstacle(),
              workspace->pde_workspace().get(),
              output_buffer)
        , params_(params)
        , workspace_(std::move(workspace))
        , grid_spacing_(workspace_->grid_spacing())
        , left_bc_(create_left_bc())
        , right_bc_(create_right_bc())
        , spatial_op_(create_spatial_op())
    {
        if (!workspace_) {
            throw std::invalid_argument("Workspace cannot be null");
        }
    }

    // CRTP interface
    const auto& left_boundary() const { return left_bc_; }
    const auto& right_boundary() const { return right_bc_; }
    const auto& spatial_operator() const { return spatial_op_; }

    // Grid info accessors
    double x_min() const {
        auto grid = workspace_->grid().subspan(0, workspace_->n_space());
        return grid[0];
    }
    double x_max() const {
        auto grid = workspace_->grid().subspan(0, workspace_->n_space());
        return grid[grid.size() - 1];
    }
    size_t n_space() const { return workspace_->n_space(); }
    size_t n_time() const { return workspace_->n_time(); }

private:
    // ... BC and operator creation same as before ...

    operators::SpatialOperator<operators::BlackScholesPDE<double>, double> create_spatial_op() const {
        auto pde = operators::BlackScholesPDE<double>(
            params_.volatility,
            params_.rate,
            params_.dividend_yield);
        return operators::create_spatial_operator(std::move(pde), grid_spacing_);
    }

    PricingParams params_;
    std::shared_ptr<AmericanSolverWorkspace> workspace_;
    GridSpacing grid_spacing_;  // Value type, not shared_ptr

    DirichletBC<LeftBCFunction> left_bc_;
    DirichletBC<RightBCFunction> right_bc_;
    operators::SpatialOperator<operators::BlackScholesPDE<double>, double> spatial_op_;
};
```

**Step 4: Update AmericanCallSolver similarly**

Apply same changes to AmericanCallSolver in the same file.

**Step 5: Run test to verify it passes**

Run: `bazel test //tests:american_option_test --test_output=errors`
Expected: PASS (at least for the new test)

**Step 6: Commit**

```bash
git add src/option/american_pde_solver.hpp tests/american_option_test.cc
git commit -m "Update American solvers for PMR workspace API

- Accept shared_ptr<AmericanSolverWorkspace>
- Store GridSpacing as value type
- Extract grid spans from workspace for PDESolver base
- Update tests to use new workspace creation"
```

---

### Task 5.2: Update Remaining American Option Tests

**Files:**
- Modify: `tests/american_option_test.cc`

**Step 1: Update test fixture**

In `tests/american_option_test.cc`, update SetUp():

```cpp
class AmericanOptionPricingTest : public ::testing::Test {
protected:
    void SetUp() override {
        pool_ = std::make_unique<std::pmr::synchronized_pool_resource>();
        auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0);

        auto workspace_result = AmericanSolverWorkspace::create(
            grid_spec.value(), 2000, pool_.get());
        ASSERT_TRUE(workspace_result.has_value()) << workspace_result.error();
        workspace_ = workspace_result.value();
    }

    std::unique_ptr<std::pmr::synchronized_pool_resource> pool_;
    std::shared_ptr<AmericanSolverWorkspace> workspace_;
};
```

**Step 2: Update Solve helper method**

```cpp
[[nodiscard]] AmericanOptionResult Solve(const AmericanOptionParams& params) const {
    return SolveWithWorkspace(params, workspace_);
}

static AmericanOptionResult SolveWithWorkspace(
    const AmericanOptionParams& params,
    const std::shared_ptr<AmericanSolverWorkspace>& workspace)
{
    auto solver_result = AmericanOptionSolver::create(params, workspace);
    if (!solver_result) {
        ADD_FAILURE() << "Failed to create solver: " << solver_result.error();
        return {};
    }

    auto solve_result = solver_result.value().solve();
    if (!solve_result) {
        const auto& error = solve_result.error();
        ADD_FAILURE() << "Solver failed: " << error.message
                      << " (code=" << static_cast<int>(error.code)
                      << ", iterations=" << error.iterations << ")";
        return {};
    }

    return solve_result.value();
}
```

**Step 3: Run all American option tests**

Run: `bazel test //tests:american_option_test --test_output=errors`
Expected: PASS for all active tests

**Step 4: Commit**

```bash
git add tests/american_option_test.cc
git commit -m "Update American option test fixture for PMR

- Create synchronized_pool_resource in SetUp
- Pass resource to workspace factory
- Update helper methods for new API"
```

---

## Phase 6: Example Updates

### Task 6.1: Update example_newton_solver.cc

**Files:**
- Modify: `examples/example_newton_solver.cc`

**Step 1: Update example to use PMR workspace**

```cpp
#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/pde_workspace_pmr.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/operators/laplacian_pde.hpp"
#include "src/pde/operators/operator_factory.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"
#include <iostream>
#include <memory_resource>
#include <cmath>

// ... TestPDESolver helper unchanged ...

int main() {
    const size_t n = 101;
    auto grid_spec_result = mango::GridSpec<double>::uniform(0.0, 1.0, n);
    if (!grid_spec_result) {
        std::cerr << "Failed to create grid spec: " << grid_spec_result.error() << "\n";
        return 1;
    }

    // Create PMR workspace
    std::pmr::synchronized_pool_resource pool;
    auto workspace_result = mango::PDEWorkspace::create(
        grid_spec_result.value(), &pool);
    if (!workspace_result.has_value()) {
        std::cerr << "Failed to create workspace: " << workspace_result.error() << "\n";
        return 1;
    }
    auto workspace = workspace_result.value();

    mango::TimeDomain time(0.0, 0.1, 0.001);

    mango::DirichletBC left_bc{[](double, double) { return 0.0; }};
    mango::DirichletBC right_bc{[](double, double) { return 0.0; }};

    auto pde = mango::operators::LaplacianPDE<double>(1.0);
    auto grid_span = workspace->grid().subspan(0, n);
    auto grid_view = mango::GridView<double>(grid_span);
    auto spatial_op = mango::operators::create_spatial_operator(std::move(pde), grid_view);

    auto solver = make_solver(grid_span, time, left_bc, right_bc, spatial_op);

    // Initial condition: u(x, 0) = sin(πx)
    auto initial_condition = [](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(M_PI * x[i]);
        }
    };
    solver.initialize(initial_condition);

    std::cout << "Solving heat equation with Newton-Raphson...\n";
    std::cout << "Grid size: " << n << "\n";
    std::cout << "Time steps: " << time.n_steps() << "\n";

    auto status = solver.solve();

    if (status) {
        std::cout << "Solver converged successfully!\n\n";
        auto solution = solver.solution();

        std::cout << "Solution at t=" << time.t_end() << ":\n";
        auto grid = workspace->grid().subspan(0, n);
        for (size_t i = 0; i < n; i += 20) {
            std::cout << "  u(" << grid[i] << ") = " << solution[i] << "\n";
        }
    } else {
        std::cout << "Solver failed to converge: " << status.error().message << "\n";
        return 1;
    }

    return 0;
}
```

**Step 2: Build and run**

Run: `bazel build //examples:example_newton_solver && ./bazel-bin/examples/example_newton_solver`
Expected: Builds and runs successfully

**Step 3: Commit**

```bash
git add examples/example_newton_solver.cc
git commit -m "Update example to use PMR workspace

- Create synchronized_pool_resource
- Use PDEWorkspace::create() factory
- Extract grid spans for solver construction"
```

---

## Phase 7: Cleanup Old Workspace

### Task 7.1: Remove Old PDEWorkspace

**Files:**
- Delete: `src/pde/core/pde_workspace.hpp`
- Rename: `src/pde/core/pde_workspace_pmr.hpp` → `src/pde/core/pde_workspace.hpp`
- Modify: `src/pde/core/BUILD.bazel`

**Step 1: Verify all tests pass with new workspace**

Run: `bazel test //...`
Expected: All tests PASS

**Step 2: Remove old workspace file**

```bash
git rm src/pde/core/pde_workspace.hpp
```

**Step 3: Rename new workspace to canonical name**

```bash
git mv src/pde/core/pde_workspace_pmr.hpp src/pde/core/pde_workspace.hpp
```

**Step 4: Update BUILD file**

In `src/pde/core/BUILD.bazel`, rename target:

```python
cc_library(
    name = "pde_workspace",  # Was: pde_workspace_pmr
    hdrs = ["pde_workspace.hpp"],
    deps = [
        ":grid",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 5: Update all includes**

```bash
# Find and replace in all files
find . -name "*.hpp" -o -name "*.cpp" -o -name "*.cc" | \
  xargs sed -i 's|pde_workspace_pmr.hpp|pde_workspace.hpp|g'
```

**Step 6: Run all tests**

Run: `bazel test //...`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add -A
git commit -m "Replace old PDEWorkspace with PMR version

- Remove raw pointer-based workspace
- Rename pde_workspace_pmr.hpp to pde_workspace.hpp
- Update all includes and BUILD rules
- All tests passing with PMR workspace"
```

---

### Task 7.2: Remove WorkspaceBase

**Files:**
- Delete: `src/support/memory/workspace_base.hpp`
- Modify: `src/support/memory/BUILD.bazel`

**Step 1: Verify WorkspaceBase no longer used**

```bash
grep -r "WorkspaceBase" src/ tests/
```

Expected: No matches (already removed in PDEWorkspace refactor)

**Step 2: Remove file**

```bash
git rm src/support/memory/workspace_base.hpp
```

**Step 3: Update BUILD**

Remove WorkspaceBase target from `src/support/memory/BUILD.bazel`

**Step 4: Run tests**

Run: `bazel test //...`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/support/memory/workspace_base.hpp src/support/memory/BUILD.bazel
git commit -m "Remove WorkspaceBase class

No longer needed with PMR-based workspace implementation.
All allocation logic moved to PDEWorkspace factory."
```

---

## Phase 8: Documentation

### Task 8.1: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update memory management section**

Add to CLAUDE.md:

```markdown
## PMR-Based Memory Management

The library uses C++17 PMR (Polymorphic Memory Resource) for efficient memory allocation in solver components.

### PDEWorkspace

**Purpose**: Unified workspace for PDE and Newton solver arrays

**Usage**:
```cpp
// Create synchronized pool for thread-safe allocation
std::pmr::synchronized_pool_resource pool;

// Create workspace from GridSpec
auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0);
auto workspace = PDEWorkspace::create(grid_spec, &pool);

// Access arrays (all return SIMD-padded spans)
auto u_current = workspace->u_current();
auto jacobian_diag = workspace->jacobian_diag();

// Extract logical size when needed
auto logical_u = u_current.subspan(0, workspace->logical_size());
```

**Key features**:
- All arrays stored in `std::pmr::vector` with caller-provided resource
- SIMD padding (round to 8 doubles) for AVX-512 safety
- Unified storage for PDE (u_current, u_next, rhs, Lu) and Newton (jacobian, residual, delta_u) arrays
- Factory pattern validates inputs and returns `expected<shared_ptr, string>`

### AmericanSolverWorkspace

**Purpose**: High-level workspace for American option pricing

**Usage**:
```cpp
std::pmr::synchronized_pool_resource pool;
auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0);

auto workspace = AmericanSolverWorkspace::create(
    grid_spec, n_time, &pool);

auto solver = AmericanPutSolver(params, workspace);
```

**Resource flow**:
- Caller creates `synchronized_pool_resource` (thread-safe)
- Passes to AmericanSolverWorkspace::create()
- Propagates to PDEWorkspace for all array allocation
- Results allocate from `get_default_resource()` (independent lifecycle)

### GridSpacing

**Value type** with span views over workspace data:
```cpp
auto workspace = PDEWorkspace::create(grid_spec, &pool).value();
auto spacing = GridSpacing::create(
    workspace->grid().subspan(0, n),
    workspace->dx().subspan(0, n-1)).value();

if (spacing.is_uniform()) {
    double dx = spacing.spacing();
} else {
    auto dx_left = spacing.dx_left_inv();
}
```

No shared_ptr needed - lightweight value type.
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: Add PMR memory management section

Document PDEWorkspace, AmericanSolverWorkspace, and GridSpacing
usage patterns with PMR resources."
```

---

## Verification

### Task V.1: Run Full Test Suite

**Step 1: Run all tests**

```bash
bazel test //... --test_output=errors
```

Expected: All tests PASS

**Step 2: Run benchmarks**

```bash
bazel build -c opt //benchmarks:readme_benchmarks
./bazel-bin/benchmarks/readme_benchmarks
```

Expected: Performance similar to baseline

**Step 3: Run examples**

```bash
bazel run //examples:example_newton_solver
```

Expected: Runs successfully

---

## Summary

This plan refactors the PDE solver memory management to use PMR throughout:

**Completed**:
1. ✅ PDEWorkspace with pmr::vector storage
2. ✅ Newton arrays merged into PDEWorkspace
3. ✅ GridSpacing as value type with span views
4. ✅ Simplified PDESolver constructor (config as member)
5. ✅ AmericanSolverWorkspace integration
6. ✅ American solver updates
7. ✅ Example updates
8. ✅ Old workspace cleanup
9. ✅ Documentation

**Benefits**:
- Thread-safe allocation with `synchronized_pool_resource`
- Clear separation of temporary (workspace) vs persistent (result) memory
- Simplified ownership with value types where appropriate
- Easier testing with injectable memory resources
- Path to global context and thread-local pools

**Next steps** (future work):
- Migrate workspace_resource to thread-local resetable pool
- Add global context for default resources
- Benchmark memory usage vs old implementation
