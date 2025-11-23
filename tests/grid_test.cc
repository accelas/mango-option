#include "src/pde/core/grid.hpp"
#include <gtest/gtest.h>

TEST(GridSpecTest, UniformGridGeneration) {
    auto result = mango::GridSpec<>::uniform(0.0, 1.0, 11);
    ASSERT_TRUE(result.has_value());
    auto grid = result.value().generate();

    EXPECT_EQ(grid.size(), 11);
    EXPECT_DOUBLE_EQ(grid[0], 0.0);
    EXPECT_DOUBLE_EQ(grid[10], 1.0);
    EXPECT_DOUBLE_EQ(grid[5], 0.5);  // Check midpoint
}

TEST(GridSpecTest, UniformGridSpacing) {
    auto result = mango::GridSpec<>::uniform(0.0, 10.0, 6);
    ASSERT_TRUE(result.has_value());
    auto grid = result.value().generate();

    // Points should be: 0, 2, 4, 6, 8, 10
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(grid[i], static_cast<double>(i * 2));
    }
}

TEST(GridSpecTest, LogSpacedGridGeneration) {
    auto result = mango::GridSpec<>::log_spaced(1.0, 100.0, 5);
    ASSERT_TRUE(result.has_value());
    auto grid = result.value().generate();

    EXPECT_EQ(grid.size(), 5);
    EXPECT_DOUBLE_EQ(grid[0], 1.0);
    EXPECT_DOUBLE_EQ(grid[4], 100.0);
    // Geometric spacing: midpoint in log space should be sqrt(1*100) = 10
    EXPECT_NEAR(grid[2], 10.0, 1e-10);
}

TEST(GridSpecTest, SinhSpacedGridGeneration) {
    auto result = mango::GridSpec<>::sinh_spaced(0.0, 1.0, 11, 2.0);
    ASSERT_TRUE(result.has_value());
    auto grid = result.value().generate();

    EXPECT_EQ(grid.size(), 11);
    EXPECT_DOUBLE_EQ(grid[0], 0.0);
    EXPECT_DOUBLE_EQ(grid[10], 1.0);
    EXPECT_DOUBLE_EQ(grid[5], 0.5);  // Center point should be at midpoint
}

TEST(GridViewTest, ViewFromBuffer) {
    auto result = mango::GridSpec<>::uniform(0.0, 10.0, 6);
    ASSERT_TRUE(result.has_value());
    auto grid = result.value().generate();
    auto view = grid.view();

    EXPECT_EQ(view.size(), 6);
    EXPECT_DOUBLE_EQ(view[0], 0.0);
    EXPECT_DOUBLE_EQ(view[5], 10.0);
}

TEST(GridViewTest, ViewIsCheapToCopy) {
    auto result = mango::GridSpec<>::uniform(0.0, 1.0, 11);
    ASSERT_TRUE(result.has_value());
    auto grid = result.value().generate();
    auto view1 = grid.view();
    auto view2 = view1;  // Copy view (cheap)

    EXPECT_EQ(view2.size(), 11);
    EXPECT_DOUBLE_EQ(view2[0], 0.0);
}

TEST(GridViewTest, ViewFromSpan) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto view = mango::GridView<>(std::span<const double>(data));

    EXPECT_EQ(view.size(), 5);
    EXPECT_DOUBLE_EQ(view[2], 3.0);
}

TEST(GridViewTest, GridBoundaryAccessors) {
    auto result = mango::GridSpec<>::uniform(0.0, 10.0, 6);
    ASSERT_TRUE(result.has_value());
    auto grid = result.value().generate();
    auto view = grid.view();

    EXPECT_DOUBLE_EQ(view.x_min(), 0.0);
    EXPECT_DOUBLE_EQ(view.x_max(), 10.0);
}

TEST(GridViewTest, UniformGridDetection) {
    auto uniform_result = mango::GridSpec<>::uniform(0.0, 1.0, 11);
    ASSERT_TRUE(uniform_result.has_value());
    auto uniform_grid = uniform_result.value().generate();
    EXPECT_TRUE(uniform_grid.view().is_uniform());

    auto log_result = mango::GridSpec<>::log_spaced(1.0, 100.0, 11);
    ASSERT_TRUE(log_result.has_value());
    auto log_grid = log_result.value().generate();
    EXPECT_FALSE(log_grid.view().is_uniform());
}

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

TEST(GridViewTest, SpanAndDataAccess) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    auto view = mango::GridView<>(std::span<const double>(data));

    auto s = view.span();
    EXPECT_EQ(s.size(), 3);
    EXPECT_EQ(s[1], 2.0);

    const double* ptr = view.data();
    EXPECT_EQ(ptr[2], 3.0);
}

TEST(GridSpecTest, MultiSinhTypeExists) {
    using Type = mango::GridSpec<>::Type;
    Type t = Type::MultiSinhSpaced;
    EXPECT_EQ(t, Type::MultiSinhSpaced);
}

TEST(GridSpecTest, ClustersStorageAccessor) {
    auto result = mango::GridSpec<>::uniform(0.0, 1.0, 11);
    ASSERT_TRUE(result.has_value());

    auto clusters = result.value().clusters();
    EXPECT_TRUE(clusters.empty());
}

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

    // Check strict monotonicity (critical for finite difference operators)
    for (size_t i = 1; i < grid.size(); ++i) {
        EXPECT_GT(grid[i], grid[i-1]) << "Grid must be strictly monotonic at index " << i;
    }
}

TEST(GridSpecTest, MultiSinhSingleClusterMatchesSinhSpaced) {
    // Single-cluster multi-sinh centered at domain midpoint should match regular sinh_spaced
    const double x_min = -3.0;
    const double x_max = 3.0;
    const double center = (x_min + x_max) / 2.0;  // 0.0
    const size_t n = 21;
    const double alpha = 2.5;

    // Generate multi-sinh grid with single cluster at center
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = center, .alpha = alpha, .weight = 1.0}
    };
    auto multi_result = mango::GridSpec<>::multi_sinh_spaced(x_min, x_max, n, clusters);
    ASSERT_TRUE(multi_result.has_value());
    auto multi_grid = multi_result.value().generate();

    // Generate regular sinh_spaced grid (always centers at midpoint)
    auto sinh_result = mango::GridSpec<>::sinh_spaced(x_min, x_max, n, alpha);
    ASSERT_TRUE(sinh_result.has_value());
    auto sinh_grid = sinh_result.value().generate();

    // Grids should match exactly
    ASSERT_EQ(multi_grid.size(), sinh_grid.size());
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(multi_grid[i], sinh_grid[i], 1e-14)
            << "Mismatch at index " << i;
    }
}

TEST(GridSpecTest, MultiSinhRejectsSingleOffCenterCluster) {
    // Test that single-cluster multi-sinh rejects off-center clusters
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = 1.0, .alpha = 2.0, .weight = 1.0}  // Off-center (domain center is 0.0)
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 11, clusters);

    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("centered cluster"), std::string::npos);
}

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

// Task 7: Validation tests for edge cases

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
