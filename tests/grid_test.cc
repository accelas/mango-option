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
