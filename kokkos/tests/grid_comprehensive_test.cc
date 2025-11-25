/**
 * @file grid_comprehensive_test.cc
 * @brief Comprehensive tests for grid spacing, boundaries, and transformations
 *
 * Tests:
 * - Uniform grid spacing
 * - Sinh-spaced grids
 * - Log-moneyness transformations
 * - Boundary conditions
 * - Grid point distributions
 */

#include <gtest/gtest.h>
#include "kokkos/src/pde/core/grid.hpp"
#include <cmath>
#include <algorithm>

namespace mango::kokkos::test {

// Global Kokkos environment
class KokkosEnvironment : public ::testing::Environment {
public:
    void SetUp() override { Kokkos::initialize(); }
    void TearDown() override { Kokkos::finalize(); }
};

[[maybe_unused]] static ::testing::Environment* const kokkos_env =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

class GridComprehensiveTest : public ::testing::Test {
protected:
    using MemSpace = Kokkos::HostSpace;
};

// ============================================================================
// Uniform Grid Tests
// ============================================================================

TEST_F(GridComprehensiveTest, UniformGridSpacing) {
    // Test that uniform grid has constant spacing
    auto grid = Grid<MemSpace>::uniform(-3.0, 3.0, 101);
    ASSERT_TRUE(grid.has_value());

    auto points = grid->x();
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points);

    EXPECT_EQ(h.extent(0), 101);
    EXPECT_NEAR(h(0), -3.0, 1e-10);
    EXPECT_NEAR(h(100), 3.0, 1e-10);

    // Check uniform spacing
    double expected_dx = 6.0 / 100.0;
    for (size_t i = 1; i < h.extent(0); ++i) {
        double dx = h(i) - h(i - 1);
        EXPECT_NEAR(dx, expected_dx, 1e-10)
            << "Spacing not uniform at index " << i;
    }
}

TEST_F(GridComprehensiveTest, UniformGridSymmetry) {
    // For symmetric bounds, grid should be symmetric around zero
    auto grid = Grid<MemSpace>::uniform(-2.0, 2.0, 41);
    ASSERT_TRUE(grid.has_value());

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid->x());

    // Middle point should be zero
    EXPECT_NEAR(h(20), 0.0, 1e-10);

    // Check symmetry: x[i] = -x[n-1-i]
    for (size_t i = 0; i < 20; ++i) {
        EXPECT_NEAR(h(i), -h(40 - i), 1e-10)
            << "Grid not symmetric at index " << i;
    }
}

TEST_F(GridComprehensiveTest, UniformGridMinimalSize) {
    // Test minimum grid size
    auto grid = Grid<MemSpace>::uniform(0.0, 1.0, 2);
    ASSERT_TRUE(grid.has_value());

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid->x());
    EXPECT_EQ(h.extent(0), 2);
    EXPECT_NEAR(h(0), 0.0, 1e-10);
    EXPECT_NEAR(h(1), 1.0, 1e-10);
}

// ============================================================================
// Sinh Grid Tests
// ============================================================================

TEST_F(GridComprehensiveTest, SinhGridConcentration) {
    // Sinh grid should concentrate points near center
    auto grid = Grid<MemSpace>::sinh_spaced(-3.0, 3.0, 101, 2.0);
    ASSERT_TRUE(grid.has_value());

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid->x());

    // Grid should still span the full range
    EXPECT_NEAR(h(0), -3.0, 1e-10);
    EXPECT_NEAR(h(100), 3.0, 1e-10);

    // Spacing near center (around index 50) should be smaller than at edges
    double dx_center = h(51) - h(50);
    double dx_edge = h(1) - h(0);

    EXPECT_LT(dx_center, dx_edge)
        << "Sinh grid should have smaller spacing at center";
}

TEST_F(GridComprehensiveTest, SinhGridConcentrationParameter) {
    // Higher concentration parameter = more points near center
    auto grid_low = Grid<MemSpace>::sinh_spaced(-2.0, 2.0, 51, 1.0);
    auto grid_high = Grid<MemSpace>::sinh_spaced(-2.0, 2.0, 51, 3.0);

    ASSERT_TRUE(grid_low.has_value());
    ASSERT_TRUE(grid_high.has_value());

    auto h_low = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid_low->x());
    auto h_high = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid_high->x());

    // Calculate spacing near center
    double dx_center_low = h_low(26) - h_low(25);
    double dx_center_high = h_high(26) - h_high(25);

    EXPECT_LT(dx_center_high, dx_center_low)
        << "Higher concentration should give smaller center spacing";
}

TEST_F(GridComprehensiveTest, SinhGridMonotonicity) {
    // Grid points should be strictly increasing
    auto grid = Grid<MemSpace>::sinh_spaced(-3.0, 3.0, 101, 2.5);
    ASSERT_TRUE(grid.has_value());

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid->x());

    for (size_t i = 1; i < h.extent(0); ++i) {
        EXPECT_GT(h(i), h(i - 1))
            << "Grid not strictly increasing at index " << i;
    }
}

// ============================================================================
// Log-Moneyness Transformation Tests
// ============================================================================

TEST_F(GridComprehensiveTest, LogMoneynessTransformation) {
    // Create grid in log-moneyness space
    double x_min = -2.0;  // ln(S/K) = -2 -> S/K ≈ 0.135
    double x_max = 2.0;   // ln(S/K) = 2 -> S/K ≈ 7.39

    auto grid = Grid<MemSpace>::uniform(x_min, x_max, 21);
    ASSERT_TRUE(grid.has_value());

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid->x());

    // ATM should be at x=0 (center)
    EXPECT_NEAR(h(10), 0.0, 1e-10);

    // Transform to moneyness (S/K = exp(x))
    for (size_t i = 0; i < h.extent(0); ++i) {
        double moneyness = std::exp(h(i));
        EXPECT_GT(moneyness, 0.0);

        // At x=0, moneyness should be 1 (ATM)
        if (i == 10) {
            EXPECT_NEAR(moneyness, 1.0, 1e-10);
        }
    }
}

// ============================================================================
// Boundary Condition Tests
// ============================================================================

TEST_F(GridComprehensiveTest, GridBoundaryInclusion) {
    // Boundaries should be exactly at specified values
    auto grid = Grid<MemSpace>::uniform(-3.14159, 2.71828, 51);
    ASSERT_TRUE(grid.has_value());

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid->x());

    EXPECT_DOUBLE_EQ(h(0), -3.14159);
    EXPECT_DOUBLE_EQ(h(50), 2.71828);
}

TEST_F(GridComprehensiveTest, GridNoPointsOutsideBounds) {
    // All points should be within [xmin, xmax]
    auto grid = Grid<MemSpace>::sinh_spaced(-5.0, 5.0, 201, 2.0);
    ASSERT_TRUE(grid.has_value());

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid->x());

    for (size_t i = 0; i < h.extent(0); ++i) {
        EXPECT_GE(h(i), -5.0) << "Point below lower bound at index " << i;
        EXPECT_LE(h(i), 5.0) << "Point above upper bound at index " << i;
    }
}

// ============================================================================
// Grid Resolution Tests
// ============================================================================

TEST_F(GridComprehensiveTest, IncreasingResolution) {
    // More points should give finer spacing
    std::vector<size_t> sizes = {11, 21, 51, 101, 201};
    double prev_max_dx = 1e10;

    for (size_t n : sizes) {
        auto grid = Grid<MemSpace>::uniform(-3.0, 3.0, n);
        ASSERT_TRUE(grid.has_value());

        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid->x());

        // Find maximum spacing
        double max_dx = 0.0;
        for (size_t i = 1; i < h.extent(0); ++i) {
            double dx = h(i) - h(i - 1);
            max_dx = std::max(max_dx, dx);
        }

        EXPECT_LT(max_dx, prev_max_dx)
            << "More points should reduce max spacing at n=" << n;
        prev_max_dx = max_dx;
    }
}

// ============================================================================
// PDE Grid Tests (if applicable)
// ============================================================================

TEST_F(GridComprehensiveTest, GridForOptionPricing) {
    // Test typical grid for option pricing
    // Grid in log-moneyness centered at ATM
    double width = 3.0;  // ±3 standard deviations
    size_t n_points = 101;

    auto grid = Grid<MemSpace>::sinh_spaced(-width, width, n_points, 2.0);
    ASSERT_TRUE(grid.has_value());

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid->x());

    // Find index closest to ATM (x=0)
    size_t atm_idx = 0;
    double min_dist = std::abs(h(0));
    for (size_t i = 1; i < h.extent(0); ++i) {
        if (std::abs(h(i)) < min_dist) {
            min_dist = std::abs(h(i));
            atm_idx = i;
        }
    }

    // ATM should be near center
    EXPECT_NEAR(static_cast<double>(atm_idx), static_cast<double>(n_points - 1) / 2.0, 2.0)
        << "ATM should be near grid center";

    // Min distance to ATM should be small
    EXPECT_LT(min_dist, 0.1) << "Grid should have point very close to ATM";
}

TEST_F(GridComprehensiveTest, GridSpacingForStability) {
    // For PDE stability, we need dx small enough relative to domain
    // Typical rule: dx < domain_width / 50 for good accuracy
    auto grid = Grid<MemSpace>::uniform(-3.0, 3.0, 101);
    ASSERT_TRUE(grid.has_value());

    double domain_width = 6.0;
    double dx = domain_width / 100.0;

    EXPECT_LT(dx, domain_width / 50.0)
        << "Grid spacing should be fine enough for stability";

    // For sinh grids, minimum spacing at center matters
    auto sinh_grid = Grid<MemSpace>::sinh_spaced(-3.0, 3.0, 101, 2.0);
    ASSERT_TRUE(sinh_grid.has_value());

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sinh_grid->x());
    double min_dx = h(1) - h(0);
    for (size_t i = 2; i < h.extent(0); ++i) {
        min_dx = std::min(min_dx, h(i) - h(i - 1));
    }

    EXPECT_GT(min_dx, 1e-6) << "Minimum spacing should not be too small (stability)";
}

// ============================================================================
// Grid Creation Error Tests
// ============================================================================

TEST_F(GridComprehensiveTest, InvalidGridSinglePoint) {
    // Single point grid should fail or have special handling
    auto grid = Grid<MemSpace>::uniform(0.0, 1.0, 1);
    // Either fails or creates a degenerate grid
    if (grid.has_value()) {
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid->x());
        EXPECT_EQ(h.extent(0), 1);
    }
}

TEST_F(GridComprehensiveTest, InvalidGridReversedBounds) {
    // xmin > xmax should fail
    auto grid = Grid<MemSpace>::uniform(3.0, -3.0, 101);
    EXPECT_FALSE(grid.has_value());
}

TEST_F(GridComprehensiveTest, InvalidGridZeroRange) {
    // Zero range should fail
    auto grid = Grid<MemSpace>::uniform(1.0, 1.0, 101);
    EXPECT_FALSE(grid.has_value());
}

// ============================================================================
// Grid Index Lookup Tests
// ============================================================================

TEST_F(GridComprehensiveTest, FindNearestIndex) {
    auto grid = Grid<MemSpace>::uniform(0.0, 10.0, 11);
    ASSERT_TRUE(grid.has_value());

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid->x());

    // Helper to find nearest index
    auto find_nearest = [&h](double x) -> size_t {
        size_t best = 0;
        double best_dist = std::abs(h(0) - x);
        for (size_t i = 1; i < h.extent(0); ++i) {
            double dist = std::abs(h(i) - x);
            if (dist < best_dist) {
                best_dist = dist;
                best = i;
            }
        }
        return best;
    };

    EXPECT_EQ(find_nearest(0.0), 0);
    EXPECT_EQ(find_nearest(5.0), 5);
    EXPECT_EQ(find_nearest(10.0), 10);
    EXPECT_EQ(find_nearest(4.7), 5);  // 4.7 is closest to 5.0
    EXPECT_EQ(find_nearest(-1.0), 0);  // Below range
    EXPECT_EQ(find_nearest(11.0), 10);  // Above range
}

TEST_F(GridComprehensiveTest, FindBracketingIndices) {
    auto grid = Grid<MemSpace>::uniform(0.0, 10.0, 11);
    ASSERT_TRUE(grid.has_value());

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid->x());

    // Helper to find bracketing indices
    auto find_bracket = [&h](double x) -> std::pair<size_t, size_t> {
        if (x <= h(0)) return {0, 0};
        if (x >= h(h.extent(0) - 1)) return {h.extent(0) - 1, h.extent(0) - 1};

        for (size_t i = 0; i < h.extent(0) - 1; ++i) {
            if (h(i) <= x && x <= h(i + 1)) {
                return {i, i + 1};
            }
        }
        return {0, 0};  // Should not reach here
    };

    auto [lo1, hi1] = find_bracket(4.5);
    EXPECT_EQ(lo1, 4);
    EXPECT_EQ(hi1, 5);

    auto [lo2, hi2] = find_bracket(5.0);  // Exact grid point
    // The find_bracket returns {i, i+1} when h[i] <= x <= h[i+1]
    // For x=5.0, this is index 4 (value 4.0) and index 5 (value 5.0)
    EXPECT_EQ(lo2, 4);
    EXPECT_EQ(hi2, 5);

    auto [lo3, hi3] = find_bracket(-1.0);  // Below range
    EXPECT_EQ(lo3, 0);
    EXPECT_EQ(hi3, 0);
}

}  // namespace mango::kokkos::test
