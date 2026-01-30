# Grid-Based American Option Result Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor American option pricing to use Grid-based results, eliminating duplicate storage and enabling snapshot collection.

**Architecture:** Three-layer separation - Grid (per-solve, owns solution + snapshots), PDEWorkspace (reusable, spans over PMR buffers), AmericanOptionSolver (creates Grid, uses workspace). Grid becomes single source of truth for PDE solutions.

**Tech Stack:** C++23, Bazel, GoogleTest, std::expected, PMR, std::span

---

## Phase 1: Grid Snapshot System

### Task 1.1: Add Snapshot Storage to Grid

**Files:**
- Modify: `src/pde/core/grid.hpp`
- Modify: `src/pde/core/grid.cpp` (if impl file exists, otherwise header-only)
- Test: `tests/grid_snapshot_test.cc` (create new file)

**Step 1: Write the failing test**

Create `tests/grid_snapshot_test.cc`:

```cpp
#include "src/pde/core/grid.hpp"
#include "src/pde/core/grid_spec.hpp"
#include "src/pde/core/time_domain.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(GridSnapshotTest, CreateWithoutSnapshots) {
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 11).value();
    auto time_domain = TimeDomain::from_n_steps(0.0, 1.0, 10);

    auto grid_result = Grid<double>::create(grid_spec, time_domain);

    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value();
    EXPECT_FALSE(grid->has_snapshots());
    EXPECT_EQ(grid->num_snapshots(), 0);
}

TEST(GridSnapshotTest, CreateWithSnapshotTimes) {
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 11).value();
    auto time_domain = TimeDomain::from_n_steps(0.0, 1.0, 10);
    std::vector<double> snapshot_times = {0.0, 0.5, 1.0};

    auto grid_result = Grid<double>::create(grid_spec, time_domain, snapshot_times);

    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value();
    EXPECT_TRUE(grid->has_snapshots());
    EXPECT_EQ(grid->num_snapshots(), 3);
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:grid_snapshot_test --test_output=all
```

Expected: Compilation failure - `has_snapshots()`, `num_snapshots()` don't exist

**Step 3: Add snapshot members to Grid class**

In `src/pde/core/grid.hpp`, add to Grid<T> class:

```cpp
private:
    std::vector<size_t> snapshot_indices_;       // State indices to record
    std::vector<double> snapshot_times_;         // Actual times (after snapping)
    std::optional<std::vector<T>> surface_history_;  // Snapshots (row-major)

    // Helper: Find snapshot index for state index
    std::optional<size_t> find_snapshot_index(size_t state_idx) const {
        auto it = std::lower_bound(snapshot_indices_.begin(),
                                   snapshot_indices_.end(),
                                   state_idx);
        if (it != snapshot_indices_.end() && *it == state_idx) {
            return std::distance(snapshot_indices_.begin(), it);
        }
        return std::nullopt;
    }

public:
    // Snapshot query API
    bool has_snapshots() const {
        return surface_history_.has_value();
    }

    size_t num_snapshots() const {
        return snapshot_indices_.size();
    }

    std::span<const T> at(size_t snapshot_idx) const {
        if (!surface_history_.has_value() || snapshot_idx >= num_snapshots()) {
            return {};
        }
        size_t offset = snapshot_idx * n_space();
        return std::span<const T>(surface_history_->data() + offset, n_space());
    }

    std::span<const double> snapshot_times() const {
        return snapshot_times_;
    }
```

**Step 4: Run test to verify it passes**

```bash
bazel test //tests:grid_snapshot_test --test_output=all
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/pde/core/grid.hpp tests/grid_snapshot_test.cc
git commit -m "feat(grid): add snapshot storage members and query API

- Add snapshot_indices_, snapshot_times_, surface_history_ members
- Implement has_snapshots(), num_snapshots(), at(), snapshot_times()
- Add find_snapshot_index() helper for state lookup
- Tests verify creation with/without snapshots"
```

---

### Task 1.2: Implement Time-to-Index Conversion

**Files:**
- Modify: `src/pde/core/grid.hpp`
- Modify: `src/pde/core/grid.cpp`
- Test: `tests/grid_snapshot_test.cc`

**Step 1: Write the failing test**

Add to `tests/grid_snapshot_test.cc`:

```cpp
TEST(GridSnapshotTest, TimeToIndexConversion) {
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 11).value();
    auto time_domain = TimeDomain::from_n_steps(0.0, 1.0, 10);  // dt = 0.1

    // Request snapshots at t=0.0, t=0.45, t=1.0
    std::vector<double> snapshot_times = {0.0, 0.45, 1.0};

    auto grid_result = Grid<double>::create(grid_spec, time_domain, snapshot_times);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value();

    // Should snap to nearest: 0.0→state0, 0.45→state5, 1.0→state10
    EXPECT_EQ(grid->num_snapshots(), 3);
    auto times = grid->snapshot_times();
    EXPECT_NEAR(times[0], 0.0, 1e-10);
    EXPECT_NEAR(times[1], 0.5, 1e-10);  // Snapped to nearest
    EXPECT_NEAR(times[2], 1.0, 1e-10);
}

TEST(GridSnapshotTest, OutOfRangeTimeRejected) {
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 11).value();
    auto time_domain = TimeDomain::from_n_steps(0.0, 1.0, 10);

    std::vector<double> bad_times = {-0.1};  // Negative time
    auto grid_result = Grid<double>::create(grid_spec, time_domain, bad_times);

    EXPECT_FALSE(grid_result.has_value());
    EXPECT_TRUE(grid_result.error().find("out of range") != std::string::npos);
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:grid_snapshot_test::TimeToIndexConversion --test_output=all
```

Expected: FAIL - conversion not implemented

**Step 3: Implement convert_times_to_indices helper**

Add to `src/pde/core/grid.cpp` (or grid.hpp if header-only):

```cpp
namespace {

std::expected<std::pair<std::vector<size_t>, std::vector<double>>, std::string>
convert_times_to_indices(std::span<const double> times,
                         const TimeDomain& time_domain) {
    const double t_start = time_domain.t_start();
    const double t_end = time_domain.t_end();
    const size_t n_steps = time_domain.n_steps();
    const double dt = time_domain.dt();

    // Validate preconditions
    if (n_steps == 0) {
        return std::unexpected("TimeDomain has zero time steps");
    }
    if (dt <= 0.0) {
        return std::unexpected(std::format(
            "Invalid TimeDomain: dt={}", dt));
    }

    std::vector<size_t> indices;
    std::vector<double> snapped_times;
    indices.reserve(times.size());
    snapped_times.reserve(times.size());

    for (double t : times) {
        // Validate range
        if (t < t_start || t > t_end) {
            return std::unexpected(std::format(
                "Snapshot time {} out of range [{}, {}]", t, t_start, t_end));
        }

        // Convert to nearest state index
        double step_exact = (t - t_start) / dt;
        size_t state_idx = static_cast<size_t>(std::floor(step_exact + 0.5));
        state_idx = std::min(state_idx, n_steps);  // Clamp to n_steps

        indices.push_back(state_idx);
        snapped_times.push_back(t_start + state_idx * dt);
    }

    // Sort and deduplicate
    std::vector<std::pair<size_t, double>> paired;
    for (size_t i = 0; i < indices.size(); ++i) {
        paired.push_back({indices[i], snapped_times[i]});
    }
    std::sort(paired.begin(), paired.end());
    paired.erase(std::unique(paired.begin(), paired.end()), paired.end());

    indices.clear();
    snapped_times.clear();
    for (const auto& [idx, time] : paired) {
        indices.push_back(idx);
        snapped_times.push_back(time);
    }

    return std::make_pair(indices, snapped_times);
}

}  // namespace
```

**Step 4: Update Grid::create to use conversion**

Modify Grid::create signature and implementation:

```cpp
static std::expected<std::shared_ptr<Grid<T>>, std::string>
create(const GridSpec<T>& grid_spec,
       const TimeDomain& time_domain,
       std::span<const double> snapshot_times = {}) {

    auto grid = std::make_shared<Grid<T>>(grid_spec, time_domain);

    if (!snapshot_times.empty()) {
        auto conversion = convert_times_to_indices(snapshot_times, time_domain);
        if (!conversion.has_value()) {
            return std::unexpected(conversion.error());
        }

        auto [indices, times] = conversion.value();
        grid->snapshot_indices_ = std::move(indices);
        grid->snapshot_times_ = std::move(times);

        // Allocate snapshot storage
        size_t total_size = grid->snapshot_indices_.size() * grid->n_space();
        grid->surface_history_ = std::vector<T>(total_size);
    }

    return grid;
}
```

**Step 5: Run tests to verify they pass**

```bash
bazel test //tests:grid_snapshot_test --test_output=all
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/pde/core/grid.hpp src/pde/core/grid.cpp tests/grid_snapshot_test.cc
git commit -m "feat(grid): implement time-to-index conversion for snapshots

- Add convert_times_to_indices() helper function
- Validates time range, converts to state indices [0, n_steps]
- Snaps to nearest grid point, deduplicates
- Update Grid::create to use conversion and allocate storage
- Tests verify conversion accuracy and out-of-range rejection"
```

---

### Task 1.3: Add Snapshot Recording API

**Files:**
- Modify: `src/pde/core/grid.hpp`
- Test: `tests/grid_snapshot_test.cc`

**Step 1: Write the failing test**

Add to `tests/grid_snapshot_test.cc`:

```cpp
TEST(GridSnapshotTest, RecordAndRetrieve) {
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 11).value();
    auto time_domain = TimeDomain::from_n_steps(0.0, 1.0, 10);
    std::vector<double> snapshot_times = {0.0, 0.5, 1.0};

    auto grid = Grid<double>::create(grid_spec, time_domain, snapshot_times).value();

    // Record snapshots at different states
    std::vector<double> state0(11, 1.0);  // All 1.0
    std::vector<double> state5(11, 2.0);  // All 2.0
    std::vector<double> state10(11, 3.0); // All 3.0

    EXPECT_TRUE(grid->should_record(0));
    grid->record(0, state0);

    EXPECT_TRUE(grid->should_record(5));
    grid->record(5, state5);

    EXPECT_TRUE(grid->should_record(10));
    grid->record(10, state10);

    // Retrieve and verify
    auto snap0 = grid->at(0);
    EXPECT_EQ(snap0.size(), 11);
    EXPECT_DOUBLE_EQ(snap0[0], 1.0);

    auto snap1 = grid->at(1);
    EXPECT_DOUBLE_EQ(snap1[0], 2.0);

    auto snap2 = grid->at(2);
    EXPECT_DOUBLE_EQ(snap2[0], 3.0);
}

TEST(GridSnapshotTest, ShouldRecordOnlyRequestedStates) {
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 11).value();
    auto time_domain = TimeDomain::from_n_steps(0.0, 1.0, 10);
    std::vector<double> snapshot_times = {0.5};  // Only middle

    auto grid = Grid<double>::create(grid_spec, time_domain, snapshot_times).value();

    EXPECT_FALSE(grid->should_record(0));
    EXPECT_TRUE(grid->should_record(5));
    EXPECT_FALSE(grid->should_record(10));
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:grid_snapshot_test::RecordAndRetrieve --test_output=all
```

Expected: FAIL - should_record(), record() don't exist

**Step 3: Implement recording API**

Add to `src/pde/core/grid.hpp`:

```cpp
public:
    // Recording API (for PDESolver)
    bool should_record(size_t state_idx) const {
        return find_snapshot_index(state_idx).has_value();
    }

    void record(size_t state_idx, std::span<const T> solution) {
        auto snap_idx = find_snapshot_index(state_idx);
        if (!snap_idx.has_value() || !surface_history_.has_value()) {
            return;  // Silently ignore if not a snapshot state
        }

        // Copy solution to snapshot storage
        size_t offset = snap_idx.value() * n_space();
        std::copy(solution.begin(), solution.end(),
                  surface_history_->begin() + offset);
    }
```

**Step 4: Run tests to verify they pass**

```bash
bazel test //tests:grid_snapshot_test --test_output=all
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/pde/core/grid.hpp tests/grid_snapshot_test.cc
git commit -m "feat(grid): add snapshot recording API

- Implement should_record(state_idx) to check if state should be saved
- Implement record(state_idx, solution) to copy solution to storage
- Uses find_snapshot_index() for fast lookup
- Tests verify recording and retrieval of multiple snapshots"
```

---

## Phase 2: PDESolver Snapshot Integration

### Task 2.1: Add Snapshot Recording to PDESolver

**Files:**
- Modify: `src/pde/core/pde_solver.hpp`
- Test: `tests/pde_solver_snapshot_test.cc` (create new file)

**Step 1: Write the failing test**

Create `tests/pde_solver_snapshot_test.cc`:

```cpp
#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/operators/laplacian_pde.hpp"
#include "src/pde/operators/operator_factory.hpp"
#include <gtest/gtest.h>

// Simple diffusion solver for testing
class DiffusionSolver : public PDESolver<DiffusionSolver> {
public:
    DiffusionSolver(std::shared_ptr<Grid<double>> grid,
                   PDEWorkspace workspace,
                   double diffusion_coeff)
        : PDESolver<DiffusionSolver>(grid, workspace)
        , grid_(grid)
        , left_bc_([] (double, double) { return 0.0; })
        , right_bc_([] (double, double) { return 0.0; })
        , pde_(diffusion_coeff)
    {
        auto spacing_ptr = std::make_shared<GridSpacing<double>>(grid->spacing());
        spatial_op_ = operators::create_spatial_operator(pde_, spacing_ptr);
    }

    const auto& left_boundary() const { return left_bc_; }
    const auto& right_boundary() const { return right_bc_; }
    const auto& spatial_operator() const { return spatial_op_; }

private:
    std::shared_ptr<Grid<double>> grid_;
    DirichletBC<std::function<double(double, double)>> left_bc_;
    DirichletBC<std::function<double(double, double)>> right_bc_;
    operators::LaplacianPDE<double> pde_;
    operators::SpatialOperator<operators::LaplacianPDE<double>, double> spatial_op_;
};

TEST(PDESolverSnapshotTest, RecordsInitialCondition) {
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 11).value();
    auto time_domain = TimeDomain::from_n_steps(0.0, 0.1, 10);
    std::vector<double> snapshot_times = {0.0};  // Just initial condition

    auto grid = Grid<double>::create(grid_spec, time_domain, snapshot_times).value();

    std::pmr::synchronized_pool_resource pool;
    auto workspace_result = create_pde_workspace(11, &pool);
    ASSERT_TRUE(workspace_result.has_value());

    DiffusionSolver solver(grid, workspace_result.value(), 0.1);

    // Set initial condition: u = sin(pi*x)
    solver.initialize([](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(M_PI * x[i]);
        }
    });

    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    // Verify initial condition was recorded
    EXPECT_TRUE(grid->has_snapshots());
    auto snap0 = grid->at(0);
    EXPECT_NEAR(snap0[5], std::sin(M_PI * 0.5), 1e-10);  // Middle point
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:pde_solver_snapshot_test --test_output=all
```

Expected: FAIL - snapshot not recorded (PDESolver doesn't call record())

**Step 3: Add initial condition recording to PDESolver**

In `src/pde/core/pde_solver.hpp`, modify solve() method:

```cpp
std::expected<void, SolverError> solve() {
    // ... existing initialization code ...

    // Record initial condition if requested
    if (grid_->should_record(0)) {
        grid_->record(0, u_current);
    }

    // Time-stepping loop
    for (size_t step = 0; step < time_.n_steps(); ++step) {
        // ... TR-BDF2 stages ...
        // ... process_temporal_events ...

        // Record snapshot AFTER events
        if (grid_->should_record(step + 1)) {
            grid_->record(step + 1, u_current);
        }

        t = t_next;
    }

    return {};
}
```

**Step 4: Run test to verify it passes**

```bash
bazel test //tests:pde_solver_snapshot_test --test_output=all
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/pde/core/pde_solver.hpp tests/pde_solver_snapshot_test.cc
git commit -m "feat(pde): add snapshot recording to PDESolver

- Record initial condition (state 0) before time loop
- Record states after TR-BDF2 + temporal events
- Uses Grid::should_record() and Grid::record() API
- Tests verify initial condition capture"
```

---

### Task 2.2: Test Snapshot Recording During Time Steps

**Files:**
- Test: `tests/pde_solver_snapshot_test.cc`

**Step 1: Write the test**

Add to `tests/pde_solver_snapshot_test.cc`:

```cpp
TEST(PDESolverSnapshotTest, RecordsMultipleSnapshots) {
    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 11).value();
    auto time_domain = TimeDomain::from_n_steps(0.0, 1.0, 10);  // dt=0.1
    std::vector<double> snapshot_times = {0.0, 0.5, 1.0};

    auto grid = Grid<double>::create(grid_spec, time_domain, snapshot_times).value();

    std::pmr::synchronized_pool_resource pool;
    auto workspace_result = create_pde_workspace(11, &pool);

    DiffusionSolver solver(grid, workspace_result.value(), 0.1);

    solver.initialize([](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(M_PI * x[i]);
        }
    });

    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    // Should have 3 snapshots
    EXPECT_EQ(grid->num_snapshots(), 3);

    // Each snapshot should have 11 points
    EXPECT_EQ(grid->at(0).size(), 11);
    EXPECT_EQ(grid->at(1).size(), 11);
    EXPECT_EQ(grid->at(2).size(), 11);

    // Solution should decay over time (diffusion)
    double initial_peak = grid->at(0)[5];
    double mid_peak = grid->at(1)[5];
    double final_peak = grid->at(2)[5];

    EXPECT_GT(initial_peak, mid_peak);
    EXPECT_GT(mid_peak, final_peak);
}
```

**Step 2: Run test to verify it passes**

```bash
bazel test //tests:pde_solver_snapshot_test::RecordsMultipleSnapshots --test_output=all
```

Expected: PASS (PDESolver already records in loop)

**Step 3: Commit**

```bash
git add tests/pde_solver_snapshot_test.cc
git commit -m "test(pde): verify snapshot recording during time steps

- Test multiple snapshots at t=0.0, 0.5, 1.0
- Verify snapshot count and sizes
- Verify solution evolution (diffusion decay)"
```

---

## Phase 3: AmericanOptionResult Wrapper

### Task 3.1: Create New AmericanOptionResult Wrapper Class

**Files:**
- Create: `src/option/american_option_result.hpp`
- Create: `src/option/american_option_result.cpp`
- Test: `tests/american_option_result_test.cc`

**Step 1: Write the failing test**

Create `tests/american_option_result_test.cc`:

```cpp
#include "src/option/american_option_result.hpp"
#include "src/pde/core/grid.hpp"
#include <gtest/gtest.h>

TEST(AmericanOptionResultTest, ValueAtSpot) {
    // Create grid with solution
    auto grid_spec = GridSpec<double>::uniform(-1.0, 1.0, 21).value();  // log-moneyness
    auto time_domain = TimeDomain::from_n_steps(0.0, 1.0, 100);
    auto grid = Grid<double>::create(grid_spec, time_domain).value();

    // Set solution: V/K = max(1 - e^x, 0) (put payoff)
    auto solution = grid->solution();
    auto x_grid = grid->x();
    for (size_t i = 0; i < x_grid.size(); ++i) {
        solution[i] = std::max(1.0 - std::exp(x_grid[i]), 0.0);
    }

    // Create params
    PricingParams params{
        .spot = 100.0,
        .strike = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::PUT
    };

    AmericanOptionResult result(grid, params);

    // value() should be value_at(spot)
    EXPECT_DOUBLE_EQ(result.value(), result.value_at(100.0));

    // At spot=100, x=ln(100/100)=0, V/K should be ~1.0
    EXPECT_NEAR(result.value(), 100.0, 5.0);  // ~100 * 1.0 = 100
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:american_option_result_test --test_output=all
```

Expected: Compilation failure - AmericanOptionResult doesn't exist

**Step 3: Create AmericanOptionResult class**

Create `src/option/american_option_result.hpp`:

```cpp
#pragma once

#include "src/pde/core/grid.hpp"
#include "src/option/option_spec.hpp"
#include "src/pde/operators/centered_difference_facade.hpp"
#include <memory>
#include <optional>

namespace mango {

class AmericanOptionResult {
public:
    AmericanOptionResult(std::shared_ptr<Grid<double>> grid,
                        const PricingParams& params)
        : grid_(grid)
        , params_(params)
    {}

    // Convenience: value at current spot
    double value() const {
        return value_at(params_.spot);
    }

    // Interpolate to arbitrary spot price
    double value_at(double spot) const {
        double x = std::log(spot / params_.strike);
        return interpolate_solution(x) * params_.strike;
    }

    // Greeks (lazy-computed, cached)
    double delta() const;
    double gamma() const;
    double theta() const;

    // Snapshot access (delegates to grid)
    bool has_snapshots() const {
        return grid_->has_snapshots();
    }

    std::span<const double> at_time(size_t snapshot_idx) const {
        return grid_->at(snapshot_idx);
    }

    size_t num_snapshots() const {
        return grid_->num_snapshots();
    }

    std::span<const double> snapshot_times() const {
        return grid_->snapshot_times();
    }

    // Direct grid access (for advanced users)
    const Grid<double>& grid() const { return *grid_; }
    std::shared_ptr<Grid<double>> grid_ptr() const { return grid_; }

private:
    std::shared_ptr<Grid<double>> grid_;
    PricingParams params_;

    // Lazy Greeks computation
    mutable std::unique_ptr<GridSpacing<double>> grid_spacing_;
    mutable std::unique_ptr<operators::CenteredDifference<double>> diff_op_;

    double interpolate_solution(double x) const;
};

}  // namespace mango
```

**Step 4: Implement interpolate_solution (linear for now)**

Create `src/option/american_option_result.cpp`:

```cpp
#include "src/option/american_option_result.hpp"

namespace mango {

double AmericanOptionResult::interpolate_solution(double x) const {
    auto x_grid = grid_->x();
    auto solution = grid_->solution();

    // Clamp to grid boundaries
    if (x <= x_grid.front()) {
        return solution.front();
    }
    if (x >= x_grid.back()) {
        return solution.back();
    }

    // Linear interpolation
    auto it = std::lower_bound(x_grid.begin(), x_grid.end(), x);
    size_t i = std::distance(x_grid.begin(), it);

    if (i == 0) i = 1;  // Shouldn't happen due to clamp

    double x0 = x_grid[i-1];
    double x1 = x_grid[i];
    double v0 = solution[i-1];
    double v1 = solution[i];

    double alpha = (x - x0) / (x1 - x0);
    return v0 + alpha * (v1 - v0);
}

}  // namespace mango
```

**Step 5: Run test to verify it passes**

```bash
bazel test //tests:american_option_result_test --test_output=all
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/option/american_option_result.hpp src/option/american_option_result.cpp tests/american_option_result_test.cc
git commit -m "feat(option): create AmericanOptionResult wrapper class

- Wraps Grid<double> + PricingParams
- Implements value(), value_at() with interpolation
- Delegates snapshot queries to Grid
- Provides grid access for advanced users
- Tests verify value_at interpolation"
```

---

### Task 3.2: Implement Greeks Computation

**Files:**
- Modify: `src/option/american_option_result.cpp`
- Test: `tests/american_option_result_test.cc`

**Step 1: Write the failing test**

Add to `tests/american_option_result_test.cc`:

```cpp
TEST(AmericanOptionResultTest, ComputeDelta) {
    // Create grid with smooth put payoff
    auto grid_spec = GridSpec<double>::uniform(-1.0, 1.0, 101).value();
    auto time_domain = TimeDomain::from_n_steps(0.0, 1.0, 100);
    auto grid = Grid<double>::create(grid_spec, time_domain).value();

    auto solution = grid->solution();
    auto x_grid = grid->x();
    for (size_t i = 0; i < x_grid.size(); ++i) {
        solution[i] = std::max(1.0 - std::exp(x_grid[i]), 0.0);
    }

    PricingParams params{
        .spot = 100.0,
        .strike = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::PUT
    };

    AmericanOptionResult result(grid, params);

    double delta = result.delta();

    // Put delta should be negative
    EXPECT_LT(delta, 0.0);
    EXPECT_GT(delta, -1.0);
}

TEST(AmericanOptionResultTest, ComputeGamma) {
    auto grid_spec = GridSpec<double>::uniform(-1.0, 1.0, 101).value();
    auto time_domain = TimeDomain::from_n_steps(0.0, 1.0, 100);
    auto grid = Grid<double>::create(grid_spec, time_domain).value();

    auto solution = grid->solution();
    auto x_grid = grid->x();
    for (size_t i = 0; i < x_grid.size(); ++i) {
        solution[i] = std::max(1.0 - std::exp(x_grid[i]), 0.0);
    }

    PricingParams params{.spot = 100.0, .strike = 100.0, ...};
    AmericanOptionResult result(grid, params);

    double gamma = result.gamma();

    // Gamma should be positive
    EXPECT_GT(gamma, 0.0);
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:american_option_result_test::ComputeDelta --test_output=all
```

Expected: FAIL - delta(), gamma() not implemented

**Step 3: Implement Greeks using CenteredDifference**

Modify `src/option/american_option_result.cpp`:

```cpp
double AmericanOptionResult::delta() const {
    // Lazy initialize operator
    if (!diff_op_) {
        grid_spacing_ = std::make_unique<GridSpacing<double>>(grid_->spacing());
        diff_op_ = std::make_unique<operators::CenteredDifference<double>>(*grid_spacing_);
    }

    // Find spot in grid
    double x_spot = std::log(params_.spot / params_.strike);
    auto x_grid = grid_->x();
    auto solution = grid_->solution();

    // Find nearest grid point
    auto it = std::lower_bound(x_grid.begin(), x_grid.end(), x_spot);
    size_t i = std::distance(x_grid.begin(), it);
    if (i >= x_grid.size()) i = x_grid.size() - 1;
    if (i == 0) i = 1;

    // Compute first derivative: dV/dx
    std::vector<double> first_deriv(x_grid.size());
    diff_op_->compute_first_derivative(solution, first_deriv, 1, x_grid.size() - 1);

    // Delta = dV/dS = (dV/dx) * (dx/dS) = (dV/dx) * (1/S)
    double dV_dx = first_deriv[i];
    return dV_dx / params_.spot;
}

double AmericanOptionResult::gamma() const {
    // Lazy initialize
    if (!diff_op_) {
        grid_spacing_ = std::make_unique<GridSpacing<double>>(grid_->spacing());
        diff_op_ = std::make_unique<operators::CenteredDifference<double>>(*grid_spacing_);
    }

    double x_spot = std::log(params_.spot / params_.strike);
    auto x_grid = grid_->x();
    auto solution = grid_->solution();

    auto it = std::lower_bound(x_grid.begin(), x_grid.end(), x_spot);
    size_t i = std::distance(x_grid.begin(), it);
    if (i >= x_grid.size()) i = x_grid.size() - 1;
    if (i == 0) i = 1;

    // Compute second derivative: d²V/dx²
    std::vector<double> second_deriv(x_grid.size());
    diff_op_->compute_second_derivative(solution, second_deriv, 1, x_grid.size() - 1);

    // Gamma = d²V/dS² = (d²V/dx²) * (1/S²) - (dV/dx) * (1/S²)
    // Simplified: Gamma ≈ d²V/dx² / S²
    double d2V_dx2 = second_deriv[i];
    return d2V_dx2 / (params_.spot * params_.spot);
}

double AmericanOptionResult::theta() const {
    // TODO: Implement theta (requires time derivative)
    return 0.0;
}
```

**Step 4: Run tests to verify they pass**

```bash
bazel test //tests:american_option_result_test --test_output=all
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/option/american_option_result.cpp tests/american_option_result_test.cc
git commit -m "feat(option): implement delta and gamma computation

- Lazy initialize CenteredDifference operator
- Delta uses first derivative: dV/dS = (dV/dx) / S
- Gamma uses second derivative: d²V/dS² ≈ d²V/dx² / S²
- Tests verify Greeks have correct signs and ranges
- Theta left as TODO (requires time derivative)"
```

---

## Phase 4: Remove AmericanSolverWorkspace

### Task 4.1: Update AmericanOptionSolver Constructor

**Files:**
- Modify: `src/option/american_option.hpp`
- Modify: `src/option/american_option.cpp`

**Step 1: Add new constructor signature**

In `src/option/american_option.hpp`, add new constructor:

```cpp
class AmericanOptionSolver {
public:
    // NEW: Direct workspace constructor (target API)
    AmericanOptionSolver(const PricingParams& params,
                        PDEWorkspace workspace,
                        std::optional<std::span<const double>> snapshot_times = std::nullopt);

    // OLD: Keep existing constructor for backwards compatibility (deprecated)
    [[deprecated("Use PDEWorkspace directly instead of AmericanSolverWorkspace")]]
    AmericanOptionSolver(const PricingParams& params,
                        std::shared_ptr<AmericanSolverWorkspace> workspace);

    // Solve returns new wrapper
    std::expected<AmericanOptionResult, SolverError> solve();

private:
    PricingParams params_;
    PDEWorkspace workspace_;
    std::vector<double> snapshot_times_;
};
```

**Step 2: Implement new constructor**

In `src/option/american_option.cpp`:

```cpp
AmericanOptionSolver::AmericanOptionSolver(
    const PricingParams& params,
    PDEWorkspace workspace,
    std::optional<std::span<const double>> snapshot_times)
    : params_(params)
    , workspace_(workspace)
{
    if (snapshot_times.has_value()) {
        snapshot_times_.assign(snapshot_times->begin(), snapshot_times->end());
    }
}
```

**Step 3: Commit**

```bash
git add src/option/american_option.hpp src/option/american_option.cpp
git commit -m "feat(option): add PDEWorkspace-based constructor

- Add constructor taking PDEWorkspace + optional snapshots
- Mark AmericanSolverWorkspace constructor as deprecated
- Store snapshot times for Grid creation in solve()
- Prepares for AmericanSolverWorkspace removal"
```

---

### Task 4.2: Implement New solve() Method

**Files:**
- Modify: `src/option/american_option.cpp`
- Test: `tests/american_option_new_api_test.cc`

**Step 1: Write the failing test**

Create `tests/american_option_new_api_test.cc`:

```cpp
#include "src/option/american_option.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include <gtest/gtest.h>

TEST(AmericanOptionNewAPITest, SolveWithPDEWorkspace) {
    PricingParams params{
        .spot = 100.0,
        .strike = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::PUT
    };

    // Create workspace
    std::pmr::synchronized_pool_resource pool;
    size_t n_space = 101;
    auto workspace_result = create_pde_workspace(n_space, &pool);
    ASSERT_TRUE(workspace_result.has_value());

    AmericanOptionSolver solver(params, workspace_result.value());
    auto result = solver.solve();

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value(), 0.0);
    EXPECT_LT(result->value(), params.strike);
}

TEST(AmericanOptionNewAPITest, SolveWithSnapshots) {
    PricingParams params{.spot = 100.0, .strike = 100.0, ...};

    std::pmr::synchronized_pool_resource pool;
    auto workspace_result = create_pde_workspace(101, &pool);

    std::vector<double> snapshot_times = {0.0, 0.5, 1.0};
    AmericanOptionSolver solver(params, workspace_result.value(), snapshot_times);

    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->has_snapshots());
    EXPECT_EQ(result->num_snapshots(), 3);
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:american_option_new_api_test --test_output=all
```

Expected: FAIL - solve() not implemented for new API

**Step 3: Implement solve() with Grid creation**

Modify `src/option/american_option.cpp`:

```cpp
std::expected<AmericanOptionResult, SolverError>
AmericanOptionSolver::solve() {
    // Estimate grid configuration
    auto [grid_spec, time_domain] = estimate_grid_for_option(params_);
    TimeDomain time_domain = TimeDomain::from_n_steps(0.0, params_.maturity, n_time);

    // Validate workspace size
    if (workspace_.size() != grid_spec.n_points()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .message = std::format(
                "Workspace size mismatch: {} != {}",
                workspace_.size(), grid_spec.n_points())
        });
    }

    // Create Grid with optional snapshots
    auto grid_result = Grid<double>::create(grid_spec, time_domain, snapshot_times_);
    if (!grid_result.has_value()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::GridCreationFailed,
            .message = grid_result.error()
        });
    }
    auto grid = grid_result.value();

    // Initialize dx in workspace
    auto dx_span = workspace_.dx();
    auto grid_points = grid->x();
    for (size_t i = 0; i < grid_points.size() - 1; ++i) {
        dx_span[i] = grid_points[i + 1] - grid_points[i];
    }

    // Create appropriate solver (put vs call)
    if (params_.type == OptionType::PUT) {
        AmericanPutSolver pde_solver(params_, grid, workspace_);
        pde_solver.initialize(AmericanPutSolver::payoff);

        auto solve_result = pde_solver.solve();
        if (!solve_result.has_value()) {
            return std::unexpected(solve_result.error());
        }
    } else {
        AmericanCallSolver pde_solver(params_, grid, workspace_);
        pde_solver.initialize(AmericanCallSolver::payoff);

        auto solve_result = pde_solver.solve();
        if (!solve_result.has_value()) {
            return std::unexpected(solve_result.error());
        }
    }

    // Wrap Grid + params → AmericanOptionResult
    return AmericanOptionResult(grid, params_);
}
```

**Step 4: Run tests to verify they pass**

```bash
bazel test //tests:american_option_new_api_test --test_output=all
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/option/american_option.cpp tests/american_option_new_api_test.cc
git commit -m "feat(option): implement solve() with Grid creation

- Creates Grid from params + workspace size
- Validates workspace size matches grid
- Initializes workspace dx from grid spacing
- Branches on put/call to select solver type
- Returns AmericanOptionResult wrapper
- Tests verify basic solving and snapshot collection"
```

---

## Phase 5: Migration and Cleanup

### Task 5.1: Deprecate AmericanSolverWorkspace

**Files:**
- Modify: `src/option/american_solver_workspace.hpp`
- Create: `docs/migration/american-option-api-migration.md`

**Step 1: Mark class as deprecated**

In `src/option/american_solver_workspace.hpp`:

```cpp
/// @deprecated Use PDEWorkspace directly. See docs/migration/american-option-api-migration.md
class [[deprecated("Use PDEWorkspace directly")]] AmericanSolverWorkspace {
    // ... existing implementation ...
};
```

**Step 2: Create migration guide**

Create `docs/migration/american-option-api-migration.md`:

```markdown
# American Option API Migration Guide

## Overview

The American option API has been refactored to use Grid-based results and direct PDEWorkspace management.

## Breaking Changes

### 1. AmericanSolverWorkspace → PDEWorkspace

**Old API:**
```cpp
auto workspace = AmericanSolverWorkspace::create(x_min, x_max, n_space, n_time);
AmericanOptionSolver solver(params, workspace);
```

**New API:**
```cpp
std::pmr::synchronized_pool_resource pool;
size_t n_space = 101;
auto workspace = create_pde_workspace(n_space, &pool).value();
AmericanOptionSolver solver(params, workspace);
```

### 2. AmericanOptionResult Structure → Wrapper Class

**Old API:**
```cpp
auto result = solver.solve();
double price = result.value;
auto surface = result.at_time(5);
```

**New API:**
```cpp
auto result = solver.solve().value();  // Returns expected
double price = result.value();  // Method call
auto surface = result.at_time(5);  // Unchanged
```

### 3. Greeks Computation

**Old API:**
```cpp
auto greeks = solver.compute_greeks();
double delta = greeks.delta;
```

**New API:**
```cpp
auto result = solver.solve().value();
double delta = result.delta();  // Lazy computed
```

## Migration Steps

1. Replace `AmericanSolverWorkspace::create` with `create_pde_workspace`
2. Manage PMR buffer lifetime explicitly
3. Update result access from struct fields to methods
4. Update Greeks access to result methods

## Timeline

- **Deprecated:** 2025-11-21
- **Removal planned:** After 2 release cycles
```

**Step 3: Commit**

```bash
git add src/option/american_solver_workspace.hpp docs/migration/american-option-api-migration.md
git commit -m "docs: deprecate AmericanSolverWorkspace

- Mark class with [[deprecated]] attribute
- Add migration guide explaining breaking changes
- Document old vs new API patterns
- Timeline for removal: 2 release cycles"
```

---

## Summary

This plan breaks down the Grid-based American option refactoring into 15 bite-sized tasks across 5 phases:

**Phase 1 (Tasks 1.1-1.3):** Add snapshot storage, time conversion, recording API to Grid
**Phase 2 (Tasks 2.1-2.2):** Integrate snapshot recording into PDESolver
**Phase 3 (Tasks 3.1-3.2):** Create AmericanOptionResult wrapper with Greeks
**Phase 4 (Tasks 4.1-4.2):** Add PDEWorkspace-based constructor and solve()
**Phase 5 (Task 5.1):** Deprecate AmericanSolverWorkspace with migration guide

Each task follows TDD: write test, verify failure, implement, verify pass, commit.

**Note:** Discrete dividend support is intentionally omitted (marked as open issue in design document). Full implementation requires cubic spline interpolation infrastructure.
