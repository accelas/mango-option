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
