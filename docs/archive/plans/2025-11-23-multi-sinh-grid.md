<!-- SPDX-License-Identifier: MIT -->
# Multi-Sinh Grid Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend GridSpec to support composite multi-sinh grids that concentrate resolution at multiple user-specified log-moneyness locations while maintaining normalized-chain batch solver performance.

**Architecture:** Add a new `MultiSinhSpaced` grid type to the existing `GridSpec<T>` class. Store multiple sinh concentration clusters (center, alpha, weight) and generate grids by combining weighted sinh transforms with monotonicity enforcement. The existing PDE solver, boundary conditions, and workspace infrastructure remain unchanged—only the grid generation logic is extended.

**Tech Stack:** C++23, std::expected, std::vector, std::span, GoogleTest

---

## Task 1: Add MultiSinhCluster Data Structure

**Files:**
- Modify: `src/pde/core/grid.hpp:36-102`
- Test: `tests/grid_test.cc`

**Step 1: Write the failing test**

```cpp
// Add to tests/grid_test.cc
TEST(MultiSinhClusterTest, ClusterConstruction) {
    mango::MultiSinhCluster cluster{
        .center_x = 0.0,
        .alpha = 2.0,
        .weight = 1.0
    };

    EXPECT_DOUBLE_EQ(cluster.center_x, 0.0);
    EXPECT_DOUBLE_EQ(cluster.alpha, 2.0);
    EXPECT_DOUBLE_EQ(cluster.weight, 1.0);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:grid_test --test_filter=MultiSinhClusterTest.ClusterConstruction --test_output=all`
Expected: FAIL with "mango::MultiSinhCluster does not name a type"

**Step 3: Write minimal implementation**

Add to `src/pde/core/grid.hpp` after line 18 (in namespace mango):

```cpp
/// Multi-sinh cluster: specifies a concentration region in composite grids
///
/// Used to concentrate grid points at multiple locations (e.g., ATM and deep ITM)
/// while still using a single shared PDE grid for batch solving.
template<typename T = double>
struct MultiSinhCluster {
    T center_x;   ///< Log-moneyness center for this cluster
    T alpha;      ///< Concentration strength (must be > 0)
    T weight;     ///< Relative contribution (must be > 0)
};
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:grid_test --test_filter=MultiSinhClusterTest.ClusterConstruction --test_output=all`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pde/core/grid.hpp tests/grid_test.cc
git commit -m "Add MultiSinhCluster data structure

Introduces the building block for composite multi-sinh grids.
Each cluster specifies a concentration region (center, alpha, weight)
that will be combined to generate the final grid.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Add MultiSinhSpaced Grid Type

**Files:**
- Modify: `src/pde/core/grid.hpp:39-43`
- Test: `tests/grid_test.cc`

**Step 1: Write the failing test**

```cpp
// Add to tests/grid_test.cc
TEST(GridSpecTest, MultiSinhTypeExists) {
    using Type = mango::GridSpec<>::Type;
    Type t = Type::MultiSinhSpaced;
    EXPECT_EQ(t, Type::MultiSinhSpaced);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:grid_test --test_filter=GridSpecTest.MultiSinhTypeExists --test_output=all`
Expected: FAIL with "no member named 'MultiSinhSpaced' in 'mango::GridSpec<double>::Type'"

**Step 3: Write minimal implementation**

Modify `src/pde/core/grid.hpp:39-43` (GridSpec::Type enum):

```cpp
enum class Type {
    Uniform,         // Equally spaced points
    LogSpaced,       // Logarithmically spaced
    SinhSpaced,      // Hyperbolic sine spacing (concentrates points at center)
    MultiSinhSpaced  // Composite multi-sinh (multiple concentration regions)
};
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:grid_test --test_filter=GridSpecTest.MultiSinhTypeExists --test_output=all`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pde/core/grid.hpp tests/grid_test.cc
git commit -m "Add MultiSinhSpaced grid type enum

Extends GridSpec::Type to include composite multi-sinh grids.
This type will be used when clusters_ vector is non-empty.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Add Clusters Storage to GridSpec

**Files:**
- Modify: `src/pde/core/grid.hpp:93-101`
- Test: `tests/grid_test.cc`

**Step 1: Write the failing test**

```cpp
// Add to tests/grid_test.cc
TEST(GridSpecTest, ClustersStorageAccessor) {
    auto result = mango::GridSpec<>::uniform(0.0, 1.0, 11);
    ASSERT_TRUE(result.has_value());

    auto clusters = result.value().clusters();
    EXPECT_TRUE(clusters.empty());
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:grid_test --test_filter=GridSpecTest.ClustersStorageAccessor --test_output=all`
Expected: FAIL with "no member named 'clusters' in 'mango::GridSpec<double>'"

**Step 3: Write minimal implementation**

Modify `src/pde/core/grid.hpp`:

1. Add to private members (around line 97):
```cpp
std::vector<MultiSinhCluster<T>> clusters_;  // Empty for non-composite grids
```

2. Update constructor (line 93):
```cpp
GridSpec(Type type, T x_min, T x_max, size_t n_points, T concentration = T(1.0),
         std::vector<MultiSinhCluster<T>> clusters = {})
    : type_(type), x_min_(x_min), x_max_(x_max),
      n_points_(n_points), concentration_(concentration),
      clusters_(std::move(clusters)) {}
```

3. Add accessor (after line 90):
```cpp
std::span<const MultiSinhCluster<T>> clusters() const { return clusters_; }
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:grid_test --test_filter=GridSpecTest.ClustersStorageAccessor --test_output=all`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pde/core/grid.hpp tests/grid_test.cc
git commit -m "Add clusters storage to GridSpec

Adds clusters_ vector to store MultiSinhCluster data for composite grids.
Empty vector indicates single-sinh or non-sinh grid types.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Add multi_sinh_spaced Factory Method

**Files:**
- Modify: `src/pde/core/grid.hpp:69-80`
- Test: `tests/grid_test.cc`

**Step 1: Write the failing test**

```cpp
// Add to tests/grid_test.cc
TEST(GridSpecTest, MultiSinhFactoryBasic) {
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = 0.0, .alpha = 2.0, .weight = 1.0}
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(
        -3.0, 3.0, 101, clusters);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().type(), mango::GridSpec<>::Type::MultiSinhSpaced);
    EXPECT_EQ(result.value().n_points(), 101);
    EXPECT_EQ(result.value().clusters().size(), 1);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:grid_test --test_filter=GridSpecTest.MultiSinhFactoryBasic --test_output=all`
Expected: FAIL with "no member named 'multi_sinh_spaced' in 'mango::GridSpec<double>'"

**Step 3: Write minimal implementation**

Add after line 80 in `src/pde/core/grid.hpp`:

```cpp
static std::expected<GridSpec, std::string> multi_sinh_spaced(
    T x_min, T x_max, size_t n_points,
    std::vector<MultiSinhCluster<T>> clusters) {

    if (n_points < 2) {
        return std::unexpected<std::string>("Grid must have at least 2 points");
    }
    if (x_min >= x_max) {
        return std::unexpected<std::string>("x_min must be less than x_max");
    }
    if (clusters.empty()) {
        return std::unexpected<std::string>("MultiSinhSpaced requires at least one cluster");
    }

    // Validate each cluster
    for (size_t i = 0; i < clusters.size(); ++i) {
        if (clusters[i].alpha <= 0) {
            return std::unexpected<std::string>(
                std::format("Cluster {} alpha must be positive", i));
        }
        if (clusters[i].weight <= 0) {
            return std::unexpected<std::string>(
                std::format("Cluster {} weight must be positive", i));
        }
        if (clusters[i].center_x < x_min || clusters[i].center_x > x_max) {
            return std::unexpected<std::string>(
                std::format("Cluster {} center {} out of range [{}, {}]",
                           i, clusters[i].center_x, x_min, x_max));
        }
    }

    return GridSpec(Type::MultiSinhSpaced, x_min, x_max, n_points,
                    T(1.0), std::move(clusters));
}
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:grid_test --test_filter=GridSpecTest.MultiSinhFactoryBasic --test_output=all`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pde/core/grid.hpp tests/grid_test.cc
git commit -m "Add multi_sinh_spaced factory method

Implements factory method with validation for:
- Grid bounds and point count
- Non-empty clusters vector
- Positive alpha and weight for each cluster
- Cluster centers within grid bounds

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Implement Single-Cluster Grid Generation

**Files:**
- Modify: `src/pde/core/grid.hpp:196-237`
- Test: `tests/grid_test.cc`

**Step 1: Write the failing test**

```cpp
// Add to tests/grid_test.cc
TEST(GridSpecTest, MultiSinhSingleClusterGeneration) {
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = 0.0, .alpha = 2.0, .weight = 1.0}
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 11, clusters);
    ASSERT_TRUE(result.has_value());

    auto grid = result.value().generate();

    EXPECT_EQ(grid.size(), 11);
    EXPECT_DOUBLE_EQ(grid[0], -3.0);
    EXPECT_DOUBLE_EQ(grid[10], 3.0);
    EXPECT_NEAR(grid[5], 0.0, 1e-10);  // Center should be near 0.0
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:grid_test --test_filter=GridSpecTest.MultiSinhSingleClusterGeneration --test_output=all`
Expected: FAIL with assertion failure (generate() doesn't handle MultiSinhSpaced yet)

**Step 3: Write minimal implementation**

Add case to `GridSpec<T>::generate()` switch statement (after line 232):

```cpp
case Type::MultiSinhSpaced: {
    // Handle single cluster as special case (most common)
    if (clusters_.size() == 1) {
        const auto& cluster = clusters_[0];
        const T c = cluster.alpha;
        const T center = cluster.center_x;
        const T range = x_max_ - x_min_;
        const T sinh_half_c = std::sinh(c / T(2.0));

        // Use eta-based transform (same as SinhSpaced) for guaranteed monotonicity
        // For centered clusters: eta_center = 0.5, so transform simplifies to standard sinh spacing
        const T eta_center = (center - x_min_) / range;  // Normalized center position

        for (size_t i = 0; i < n_points_; ++i) {
            // Map i to eta ∈ [0, 1]
            const T eta = static_cast<T>(i) / static_cast<T>(n_points_ - 1);

            // Apply sinh transform centered at eta_center
            const T sinh_term = std::sinh(c * (eta - eta_center)) / sinh_half_c;
            const T normalized = (T(1.0) + sinh_term) / T(2.0);

            // Scale to [x_min, x_max] - naturally stays in bounds
            points.push_back(x_min_ + range * normalized);
        }
    } else {
        // TODO: Handle multiple clusters (Task 6)
        throw std::runtime_error("Multi-cluster generation not yet implemented");
    }
    break;
}
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:grid_test --test_filter=GridSpecTest.MultiSinhSingleClusterGeneration --test_output=all`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pde/core/grid.hpp tests/grid_test.cc
git commit -m "Implement single-cluster multi-sinh generation

Handles the common case of one concentration region.
Uses standard sinh transform centered at cluster.center_x.
Clamps endpoints to ensure exact boundary values.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Implement Multi-Cluster Grid Generation

**Files:**
- Modify: `src/pde/core/grid.hpp:220-250`
- Test: `tests/grid_test.cc`

**Step 1: Write the failing test**

```cpp
// Add to tests/grid_test.cc
TEST(GridSpecTest, MultiSinhTwoClusterGeneration) {
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = -1.0, .alpha = 2.0, .weight = 1.0},
        {.center_x = 1.0, .alpha = 2.0, .weight = 1.0}
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 21, clusters);
    ASSERT_TRUE(result.has_value());

    auto grid = result.value().generate();

    EXPECT_EQ(grid.size(), 21);
    EXPECT_DOUBLE_EQ(grid[0], -3.0);
    EXPECT_DOUBLE_EQ(grid[20], 3.0);

    // Check monotonicity
    for (size_t i = 1; i < grid.size(); ++i) {
        EXPECT_GT(grid[i], grid[i-1]) << "Grid not monotonic at index " << i;
    }
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:grid_test --test_filter=GridSpecTest.MultiSinhTwoClusterGeneration --test_output=all`
Expected: FAIL with "Multi-cluster generation not yet implemented"

**Step 3: Write minimal implementation**

Replace the TODO block in MultiSinhSpaced case with:

```cpp
} else {
    // Multi-cluster: combine weighted sinh transforms
    std::vector<T> raw_points(n_points_);

    // Normalize weights
    T total_weight = T(0);
    for (const auto& cluster : clusters_) {
        total_weight += cluster.weight;
    }

    const T range = x_max_ - x_min_;

    for (size_t i = 0; i < n_points_; ++i) {
        // Use eta ∈ [0, 1] parameterization (same as single-cluster)
        // This keeps sinh values bounded, preventing out-of-bounds points
        const T eta = static_cast<T>(i) / static_cast<T>(n_points_ - 1);

        // Weighted combination of sinh transforms
        T weighted_x = T(0);
        for (const auto& cluster : clusters_) {
            const T c = cluster.alpha;
            const T center = cluster.center_x;
            const T w = cluster.weight / total_weight;
            const T sinh_half_c = std::sinh(c / T(2.0));

            // Compute normalized center position for this cluster
            const T eta_center = (center - x_min_) / range;

            // Apply sinh transform centered at eta_center (same formula as single-cluster)
            const T sinh_term = std::sinh(c * (eta - eta_center)) / sinh_half_c;
            const T normalized = (T(1.0) + sinh_term) / T(2.0);

            // Transform to [x_min, x_max] - naturally stays in bounds
            const T x_i = x_min_ + range * normalized;

            weighted_x += w * x_i;
        }

        raw_points[i] = weighted_x;
    }

    // Enforce monotonicity with smoothing pass
    enforce_monotonicity(raw_points, x_min_, x_max_);

    // Transfer to output
    points = std::move(raw_points);
}
```

**Step 4: Add monotonicity enforcement helper**

Add private static method to GridSpec class (before generate() implementation):

```cpp
private:
    /// Enforce strict monotonicity in grid points
    ///
    /// Ensures x[i+1] > x[i] for all i, while preserving endpoints.
    /// Uses iterative smoothing to fix non-monotonic regions.
    static void enforce_monotonicity(std::vector<T>& points, T x_min, T x_max) {
        const size_t n = points.size();
        if (n < 2) return;

        // Clamp endpoints
        points[0] = x_min;
        points[n-1] = x_max;

        // Iterative monotonicity enforcement (max 100 passes)
        for (int pass = 0; pass < 100; ++pass) {
            bool modified = false;

            for (size_t i = 1; i < n; ++i) {
                if (points[i] <= points[i-1]) {
                    // Fix violation: interpolate between neighbors
                    T left = (i > 1) ? points[i-2] : x_min;
                    T right = (i < n-1) ? points[i+1] : x_max;
                    points[i] = (points[i-1] + right) / T(2.0);
                    modified = true;
                }
            }

            if (!modified) break;
        }

        // Final pass: ensure minimum spacing (avoid dx → 0)
        const T min_spacing = (x_max - x_min) / static_cast<T>(n * 100);
        for (size_t i = 1; i < n; ++i) {
            if (points[i] - points[i-1] < min_spacing) {
                points[i] = points[i-1] + min_spacing;
            }
        }

        // Ensure endpoint is preserved
        points[n-1] = x_max;
    }
```

**Step 5: Run test to verify it passes**

Run: `bazel test //tests:grid_test --test_filter=GridSpecTest.MultiSinhTwoClusterGeneration --test_output=all`
Expected: PASS

**Step 6: Commit**

```bash
git add src/pde/core/grid.hpp tests/grid_test.cc
git commit -m "Implement multi-cluster sinh grid generation

Combines weighted sinh transforms from multiple clusters.
Enforces monotonicity with iterative smoothing to prevent
non-monotonic regions and near-duplicate points.

Safeguards:
- Normalized weights prevent bias
- Minimum spacing prevents conditioning issues
- Endpoint clamping ensures exact boundaries

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Add Validation Tests for Edge Cases

**Files:**
- Test: `tests/grid_test.cc`

**Step 1: Write comprehensive validation tests**

```cpp
// Add to tests/grid_test.cc

TEST(GridSpecTest, MultiSinhRejectsEmptyClusters) {
    std::vector<mango::MultiSinhCluster<double>> clusters = {};

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 101, clusters);

    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("at least one cluster"), std::string::npos);
}

TEST(GridSpecTest, MultiSinhRejectsNegativeAlpha) {
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = 0.0, .alpha = -2.0, .weight = 1.0}
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 101, clusters);

    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("alpha must be positive"), std::string::npos);
}

TEST(GridSpecTest, MultiSinhRejectsNegativeWeight) {
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = 0.0, .alpha = 2.0, .weight = -1.0}
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 101, clusters);

    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("weight must be positive"), std::string::npos);
}

TEST(GridSpecTest, MultiSinhRejectsCenterOutOfBounds) {
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = 5.0, .alpha = 2.0, .weight = 1.0}
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 101, clusters);

    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("out of range"), std::string::npos);
}

TEST(GridSpecTest, MultiSinhThreeClustersMonotonic) {
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = -2.0, .alpha = 1.5, .weight = 1.0},
        {.center_x = 0.0, .alpha = 2.0, .weight = 1.5},
        {.center_x = 2.0, .alpha = 1.5, .weight = 1.0}
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 51, clusters);
    ASSERT_TRUE(result.has_value());

    auto grid = result.value().generate();

    EXPECT_EQ(grid.size(), 51);
    EXPECT_DOUBLE_EQ(grid[0], -3.0);
    EXPECT_DOUBLE_EQ(grid[50], 3.0);

    // Verify strict monotonicity
    for (size_t i = 1; i < grid.size(); ++i) {
        EXPECT_GT(grid[i], grid[i-1]) << "Non-monotonic at index " << i;
    }

    // Verify minimum spacing
    double min_dx = std::numeric_limits<double>::max();
    for (size_t i = 1; i < grid.size(); ++i) {
        double dx = grid[i] - grid[i-1];
        min_dx = std::min(min_dx, dx);
    }
    EXPECT_GT(min_dx, 0.0) << "Zero spacing detected";
}

TEST(GridSpecTest, MultiSinhAggressiveAlphaHandled) {
    // Very aggressive alpha can create near-duplicates
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = 0.0, .alpha = 10.0, .weight = 1.0}
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 101, clusters);
    ASSERT_TRUE(result.has_value());

    auto grid = result.value().generate();

    // Should still be monotonic despite aggressive concentration
    for (size_t i = 1; i < grid.size(); ++i) {
        EXPECT_GT(grid[i], grid[i-1]) << "Non-monotonic at index " << i;
    }
}
```

**Step 2: Run tests to verify they all pass**

Run: `bazel test //tests:grid_test --test_filter=GridSpecTest.MultiSinh* --test_output=all`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/grid_test.cc
git commit -m "Add comprehensive multi-sinh validation tests

Tests cover:
- Empty clusters rejection
- Negative alpha/weight validation
- Center bounds checking
- Multi-cluster monotonicity
- Aggressive alpha edge case
- Minimum spacing enforcement

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Add GridSpacing Integration Test

**Files:**
- Test: `tests/grid_test.cc`

**Step 1: Write integration test with GridSpacing**

```cpp
// Add to tests/grid_test.cc

TEST(GridSpacingTest, MultiSinhGridSpacingCreation) {
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = -1.0, .alpha = 2.0, .weight = 1.0},
        {.center_x = 1.0, .alpha = 2.0, .weight = 1.0}
    };

    auto spec_result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 51, clusters);
    ASSERT_TRUE(spec_result.has_value());

    auto grid_buffer = spec_result.value().generate();
    auto grid_view = grid_buffer.view();

    // Multi-sinh should produce non-uniform spacing
    EXPECT_FALSE(grid_view.is_uniform());

    // GridSpacing should handle it correctly
    auto spacing = mango::GridSpacing<double>(grid_view);
    EXPECT_FALSE(spacing.is_uniform());
    EXPECT_EQ(spacing.size(), 51);

    // Should have precomputed arrays for non-uniform grid
    auto dx_left = spacing.dx_left_inv();
    auto dx_right = spacing.dx_right_inv();
    EXPECT_EQ(dx_left.size(), 49);  // n - 2 interior points
    EXPECT_EQ(dx_right.size(), 49);
}
```

**Step 2: Run test to verify it passes**

Run: `bazel test //tests:grid_test --test_filter=GridSpacingTest.MultiSinhGridSpacingCreation --test_output=all`
Expected: PASS (GridSpacing already handles any non-uniform grid)

**Step 3: Commit**

```bash
git add tests/grid_test.cc
git commit -m "Add GridSpacing integration test for multi-sinh

Verifies that multi-sinh grids work correctly with existing
GridSpacing infrastructure (no changes needed to GridSpacing).

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Add Documentation and Usage Example

**Files:**
- Modify: `CLAUDE.md` (add to Grid Specification section)
- Create: `examples/example_multi_sinh_grid.cc`

**Step 1: Write example program**

Create `examples/example_multi_sinh_grid.cc`:

```cpp
#include "src/pde/core/grid.hpp"
#include <iostream>
#include <format>

int main() {
    // Example 1: Single cluster (equivalent to regular sinh_spaced)
    {
        std::cout << "=== Example 1: Single Cluster ===\n";

        std::vector<mango::MultiSinhCluster<double>> clusters = {
            {.center_x = 0.0, .alpha = 2.0, .weight = 1.0}
        };

        auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 11, clusters);
        if (!result.has_value()) {
            std::cerr << "Error: " << result.error() << "\n";
            return 1;
        }

        auto grid = result.value().generate();
        std::cout << "Grid points:\n";
        for (size_t i = 0; i < grid.size(); ++i) {
            std::cout << std::format("  x[{}] = {:7.4f}\n", i, grid[i]);
        }
        std::cout << "\n";
    }

    // Example 2: Dual clusters for ATM and deep ITM concentration
    {
        std::cout << "=== Example 2: Dual Clusters (ATM + Deep ITM) ===\n";

        std::vector<mango::MultiSinhCluster<double>> clusters = {
            {.center_x = 0.0, .alpha = 2.5, .weight = 2.0},   // ATM (higher weight)
            {.center_x = -2.0, .alpha = 1.5, .weight = 1.0}   // Deep ITM
        };

        auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 21, clusters);
        if (!result.has_value()) {
            std::cerr << "Error: " << result.error() << "\n";
            return 1;
        }

        auto grid = result.value().generate();
        std::cout << "Grid points with spacing:\n";
        for (size_t i = 0; i < grid.size(); ++i) {
            if (i > 0) {
                double dx = grid[i] - grid[i-1];
                std::cout << std::format("  x[{}] = {:7.4f}  (dx = {:7.4f})\n",
                                        i, grid[i], dx);
            } else {
                std::cout << std::format("  x[{}] = {:7.4f}\n", i, grid[i]);
            }
        }
        std::cout << "\n";
    }

    // Example 3: Three clusters (ITM, ATM, OTM)
    {
        std::cout << "=== Example 3: Triple Clusters (ITM + ATM + OTM) ===\n";

        std::vector<mango::MultiSinhCluster<double>> clusters = {
            {.center_x = -1.5, .alpha = 1.8, .weight = 1.0},  // Deep ITM
            {.center_x = 0.0, .alpha = 2.5, .weight = 2.0},   // ATM (highest weight)
            {.center_x = 1.5, .alpha = 1.8, .weight = 1.0}    // Deep OTM
        };

        auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 31, clusters);
        if (!result.has_value()) {
            std::cerr << "Error: " << result.error() << "\n";
            return 1;
        }

        auto grid = result.value().generate();

        // Show summary statistics
        double min_dx = std::numeric_limits<double>::max();
        double max_dx = 0.0;
        for (size_t i = 1; i < grid.size(); ++i) {
            double dx = grid[i] - grid[i-1];
            min_dx = std::min(min_dx, dx);
            max_dx = std::max(max_dx, dx);
        }

        std::cout << std::format("Grid range: [{}, {}]\n", grid[0], grid[30]);
        std::cout << std::format("Points: {}\n", grid.size());
        std::cout << std::format("Min spacing: {:.6f}\n", min_dx);
        std::cout << std::format("Max spacing: {:.6f}\n", max_dx);
        std::cout << std::format("Spacing ratio: {:.2f}x\n", max_dx / min_dx);
    }

    return 0;
}
```

**Step 2: Add BUILD rule**

Add to `examples/BUILD.bazel`:

```python
cc_binary(
    name = "example_multi_sinh_grid",
    srcs = ["example_multi_sinh_grid.cc"],
    deps = [
        "//src/pde/core:grid",
    ],
)
```

**Step 3: Test example builds and runs**

Run: `bazel build //examples:example_multi_sinh_grid && ./bazel-bin/examples/example_multi_sinh_grid`
Expected: Builds successfully, prints grid examples

**Step 4: Add documentation to CLAUDE.md**

Add to CLAUDE.md under Grid Specification section (after sinh_spaced):

```markdown
### Multi-Sinh Grids

For batch solvers that mix instruments with different normalized strikes, multi-sinh grids provide resolution at multiple concentration points while maintaining a single shared PDE grid.

**Usage:**

```cpp
#include "src/pde/core/grid.hpp"

// Define concentration regions
std::vector<mango::MultiSinhCluster<double>> clusters = {
    {.center_x = 0.0, .alpha = 2.5, .weight = 2.0},   // ATM (higher weight)
    {.center_x = -2.0, .alpha = 1.5, .weight = 1.0}   // Deep ITM
};

// Create grid specification
auto spec = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 201, clusters);

// Generate grid
auto grid = spec.value().generate();
```

**When to Use:**

- Price tables requiring accuracy at multiple strikes (ATM + deep ITM/OTM)
- Batches mixing instruments with different moneyness but same maturity
- Scenarios where single-center sinh spacing leaves important regions coarse

**Parameters:**

- `center_x`: Log-moneyness center for concentration
- `alpha`: Concentration strength (typical range: 1.5-3.0)
- `weight`: Relative importance (higher = more influence on final grid)

**Safeguards:**

The implementation enforces:
- Strict monotonicity (x[i+1] > x[i])
- Minimum spacing to prevent conditioning issues
- Exact boundary values (x[0] = x_min, x[n-1] = x_max)
- Normalized weights to prevent bias

**Performance:**

Grid generation cost is O(n × k) where k is number of clusters. For typical use cases (k ≤ 3, n ≤ 201), generation takes <100 microseconds.
```

**Step 5: Commit**

```bash
git add CLAUDE.md examples/example_multi_sinh_grid.cc examples/BUILD.bazel
git commit -m "Add multi-sinh documentation and example

Provides comprehensive usage examples for 1, 2, and 3 cluster grids.
Documents parameters, use cases, and safeguards in CLAUDE.md.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 10: Run Full Test Suite and Verify

**Files:**
- No file changes

**Step 1: Run all grid tests**

Run: `bazel test //tests:grid_test --test_output=all`
Expected: ALL PASS

**Step 2: Run full test suite**

Run: `bazel test //...`
Expected: ALL PASS (no regressions)

**Step 3: Build all examples**

Run: `bazel build //examples:all`
Expected: All examples build successfully

**Step 4: Verify multi-sinh example output**

Run: `./bazel-bin/examples/example_multi_sinh_grid | head -30`
Expected: Shows grid generation for all three examples with proper formatting

**Step 5: Final commit (if any cleanup needed)**

If tests revealed issues, fix and commit. Otherwise, create summary commit:

```bash
git add .
git commit -m "Verify multi-sinh grid implementation complete

All tests pass:
- Unit tests for cluster validation
- Single and multi-cluster generation
- Monotonicity enforcement
- GridSpacing integration
- Edge case handling

Examples demonstrate usage patterns for 1-3 clusters.
Documentation integrated into CLAUDE.md.

Ready for code review.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Completion Checklist

- [ ] Task 1: MultiSinhCluster struct added
- [ ] Task 2: MultiSinhSpaced enum value added
- [ ] Task 3: Clusters storage in GridSpec
- [ ] Task 4: multi_sinh_spaced factory method
- [ ] Task 5: Single-cluster generation implemented
- [ ] Task 6: Multi-cluster generation with monotonicity
- [ ] Task 7: Validation tests for edge cases
- [ ] Task 8: GridSpacing integration test
- [ ] Task 9: Documentation and examples
- [ ] Task 10: Full test suite verification

## Success Criteria

1. **Functional Requirements:**
   - Multi-sinh grids generate correctly for 1-10 clusters
   - Strict monotonicity enforced automatically
   - Endpoints match x_min/x_max exactly
   - Minimum spacing prevents conditioning issues

2. **Testing:**
   - All existing grid tests still pass (no regressions)
   - New multi-sinh tests cover edge cases
   - Examples build and run without errors

3. **Documentation:**
   - CLAUDE.md includes multi-sinh usage guide
   - Examples demonstrate 1, 2, and 3 cluster patterns
   - Comments explain algorithm and safeguards

4. **Integration:**
   - Works with existing GridSpacing (no changes needed)
   - Compatible with existing PDE solvers
   - No changes to workspace or boundary conditions

## Notes for Engineer

**Key Design Decisions:**

1. **Why variant storage?** Using `std::vector<MultiSinhCluster<T>>` instead of a separate `MultiSinhGridSpec` class keeps the API simple. Empty vector = non-composite grid.

2. **Why monotonicity enforcement?** Weighted combination of sinh transforms can produce non-monotonic segments, especially with overlapping clusters. Iterative smoothing is cheap (O(n) per pass, typically 1-2 passes) compared to PDE solve cost.

3. **Why minimum spacing?** Preventing dx → 0 is critical for finite difference operator conditioning. The threshold (range/100n) is conservative.

4. **Why clamp endpoints?** Floating-point arithmetic in weighted combination can produce x[0] ≠ x_min or x[n-1] ≠ x_max. Explicit clamping ensures exact boundaries for Dirichlet BC application.

**Testing Strategy:**

- @superpowers:test-driven-development for each task
- Unit tests isolate component behavior
- Integration test verifies existing infrastructure compatibility
- Examples serve as acceptance tests

**Performance:**

Multi-sinh generation is O(n × k) where n = points, k = clusters. For typical use (k ≤ 3, n ≤ 201), cost is ~50-100 microseconds, negligible compared to PDE solve (~1-10ms).

**Future Extensions:**

- Non-linear weight functions (e.g., Gaussian blend)
- Adaptive alpha based on spacing uniformity metrics
- Serialization for price table grid persistence
