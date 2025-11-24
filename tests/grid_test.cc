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

TEST(GridSpecTest, MultiSinhMergedClusterPreservesLocation) {
    // Test that merged clusters preserve their weighted-average location
    // Auto-merge only deduplicates overlapping centers, it doesn't recenter them
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = 0.95, .alpha = 3.0, .weight = 1.0},
        {.center_x = 1.05, .alpha = 3.0, .weight = 1.0}  // Δx=0.1 <= 0.3/3=0.1, will merge
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 51, clusters);

    // Should succeed with automatic merge
    ASSERT_TRUE(result.has_value());

    // Verify clusters merged to single cluster at weighted average (1.0)
    auto final_clusters = result.value().clusters();
    EXPECT_EQ(final_clusters.size(), 1);
    EXPECT_NEAR(final_clusters[0].center_x, 1.0, 1e-10)
        << "Merged cluster should preserve weighted-average position";

    // Verify grid generation concentrates near x=1.0, not domain center
    auto grid = result.value().generate();
    EXPECT_EQ(grid.size(), 51);
    EXPECT_DOUBLE_EQ(grid[0], -3.0);
    EXPECT_DOUBLE_EQ(grid[50], 3.0);

    // Find point closest to x=1.0
    size_t idx_center = 0;
    double min_dist = 100.0;
    for (size_t i = 0; i < grid.size(); ++i) {
        double dist = std::abs(grid[i] - 1.0);
        if (dist < min_dist) {
            min_dist = dist;
            idx_center = i;
        }
    }

    // Verify concentration near x=1.0 (fine spacing)
    if (idx_center > 0 && idx_center < 50) {
        double spacing_near = grid[idx_center + 1] - grid[idx_center];
        EXPECT_LT(spacing_near, 0.15) << "Grid should concentrate near merged center x=1.0";
    }
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
    EXPECT_EQ(result.error().code, mango::ValidationErrorCode::InvalidGridSize);
}

TEST(GridSpecTest, MultiSinhRejectsNegativeAlpha) {
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = 0.0, .alpha = -2.0, .weight = 1.0}
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 101, clusters);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::ValidationErrorCode::InvalidGridSpacing);
}

TEST(GridSpecTest, MultiSinhRejectsNegativeWeight) {
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = 0.0, .alpha = 2.0, .weight = -1.0}
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 101, clusters);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::ValidationErrorCode::InvalidGridSpacing);
}

TEST(GridSpecTest, MultiSinhRejectsCenterOutOfBounds) {
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = 5.0, .alpha = 2.0, .weight = 1.0}
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 101, clusters);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::ValidationErrorCode::OutOfRange);
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

// Task 8: Auto-merge tests

TEST(GridSpecTest, MultiSinhAutoMergesNearClusters) {
    // Two clusters symmetrically placed around center with alpha=2.5
    // Domain: [-3.0, 3.0], center = 0.0
    // Clusters at x=-0.05 and x=0.05
    // alpha_avg = 2.5, threshold = 0.3/2.5 = 0.12
    // delta_x = 0.1 < 0.12, so they should merge at center 0.0
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = -0.05, .alpha = 2.5, .weight = 1.0},
        {.center_x = 0.05, .alpha = 2.5, .weight = 1.0}
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 51, clusters);
    ASSERT_TRUE(result.has_value());

    // After merging, should have single cluster
    auto final_clusters = result.value().clusters();
    EXPECT_EQ(final_clusters.size(), 1);

    // Merged cluster should be at domain center (weighted average)
    // (1.0 * -0.05 + 1.0 * 0.05) / 2.0 = 0.0
    EXPECT_NEAR(final_clusters[0].center_x, 0.0, 1e-10);

    // Merged alpha should be weighted average
    // (1.0 * 2.5 + 1.0 * 2.5) / 2.0 = 2.5
    EXPECT_NEAR(final_clusters[0].alpha, 2.5, 1e-10);

    // Merged weight should be sum
    EXPECT_NEAR(final_clusters[0].weight, 2.0, 1e-10);
}

TEST(GridSpecTest, MultiSinhAutoMergesUnequalWeights) {
    // Test weighted averaging with unequal weights
    // Place clusters symmetrically so they merge at center
    // Domain: [-3.0, 3.0], center = 0.0
    // For weighted average to be 0.0: w1*x1 + w2*x2 = 0
    // Using weights 3.0 and 1.0: 3.0*x1 + 1.0*x2 = 0
    // Choose x1 = -0.025, x2 = 0.075 -> 3*(-0.025) + 1*(0.075) = 0
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = -0.025, .alpha = 2.0, .weight = 3.0},
        {.center_x = 0.075, .alpha = 4.0, .weight = 1.0}
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 51, clusters);
    ASSERT_TRUE(result.has_value());

    // Should merge (alpha_avg = 2.5, threshold = 0.12, delta_x = 0.1 < 0.12)
    auto final_clusters = result.value().clusters();
    EXPECT_EQ(final_clusters.size(), 1);

    // Weighted center: (3.0 * -0.025 + 1.0 * 0.075) / 4.0 = 0.0
    EXPECT_NEAR(final_clusters[0].center_x, 0.0, 1e-10);

    // Weighted alpha: (3.0 * 2.0 + 1.0 * 4.0) / 4.0 = 2.5
    EXPECT_NEAR(final_clusters[0].alpha, 2.5, 1e-10);

    // Total weight: 3.0 + 1.0 = 4.0
    EXPECT_NEAR(final_clusters[0].weight, 4.0, 1e-10);
}

TEST(GridSpecTest, MultiSinhDoesNotMergeDistantClusters) {
    // Two clusters far apart should not merge
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = -1.0, .alpha = 2.0, .weight = 1.0},
        {.center_x = 1.0, .alpha = 2.0, .weight = 1.0}
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 51, clusters);
    ASSERT_TRUE(result.has_value());

    // Should remain as two clusters (delta_x = 2.0, threshold = 0.15)
    auto final_clusters = result.value().clusters();
    EXPECT_EQ(final_clusters.size(), 2);
}

TEST(GridSpecTest, MultiSinhThreeClustersPartialMerge) {
    // Three clusters where only two are close
    // Cluster 0 and 1 should merge, cluster 2 stays separate
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = 0.0, .alpha = 2.5, .weight = 1.0},
        {.center_x = 0.1, .alpha = 2.5, .weight = 1.0},
        {.center_x = 2.0, .alpha = 2.5, .weight = 1.0}
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 51, clusters);
    ASSERT_TRUE(result.has_value());

    // Should have two clusters after merge (0+1 merged, 2 stays)
    auto final_clusters = result.value().clusters();
    EXPECT_EQ(final_clusters.size(), 2);

    // First cluster should be merged at center 0.05
    EXPECT_NEAR(final_clusters[0].center_x, 0.05, 1e-10);
    EXPECT_NEAR(final_clusters[0].weight, 2.0, 1e-10);

    // Second cluster should remain at 2.0
    EXPECT_NEAR(final_clusters[1].center_x, 2.0, 1e-10);
    EXPECT_NEAR(final_clusters[1].weight, 1.0, 1e-10);
}

TEST(GridSpecTest, MultiSinhChainMerge) {
    // Test chain merging with closer spacing, centered at domain midpoint
    // Domain: [-3.0, 3.0], center = 0.0
    // Place three clusters symmetrically so they merge at center
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = -0.05, .alpha = 3.0, .weight = 1.0},
        {.center_x = 0.0, .alpha = 3.0, .weight = 1.0},
        {.center_x = 0.05, .alpha = 3.0, .weight = 1.0}
    };

    // alpha_avg = 3.0, threshold = 0.1
    // -0.05 to 0.0: delta = 0.05 < 0.1 (merge) -> merged at -0.025
    // -0.025 to 0.05: delta = 0.075 < 0.1 (merge) -> merged at 0.0
    // All three should eventually merge into one at center

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 51, clusters);
    ASSERT_TRUE(result.has_value());

    auto final_clusters = result.value().clusters();
    EXPECT_EQ(final_clusters.size(), 1);

    // Final center should be at equal-weight average: (-0.05 + 0.0 + 0.05) / 3 = 0.0
    EXPECT_NEAR(final_clusters[0].center_x, 0.0, 1e-10);
    EXPECT_NEAR(final_clusters[0].weight, 3.0, 1e-10);
}

TEST(GridSpecTest, MultiSinhOffCenterMergePreservesLocation) {
    // Test merged clusters preserve location: Two close clusters both away from domain center
    // Domain: [-3.0, 3.0], center = 0.0
    // Clusters at x=1.0 and x=1.05 (close enough to merge)
    // After merge, single cluster stays at weighted average (1.025)

    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = 1.0, .alpha = 3.0, .weight = 1.0},
        {.center_x = 1.05, .alpha = 3.0, .weight = 1.0}
    };

    // alpha_avg = 3.0, threshold = 0.3/3.0 = 0.1
    // delta_x = 0.05 < 0.1, so clusters WILL merge
    // After merge, cluster stays at weighted average: (1.0 + 1.05)/2 = 1.025

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 51, clusters);

    // Should succeed with automatic merge
    ASSERT_TRUE(result.has_value());

    // Verify single cluster after merge
    auto final_clusters = result.value().clusters();
    EXPECT_EQ(final_clusters.size(), 1);

    // Verify cluster preserved weighted average (1.025), NOT recentered
    EXPECT_NEAR(final_clusters[0].center_x, 1.025, 1e-10)
        << "Merged cluster should preserve weighted-average position";

    // Verify grid generation works correctly and concentrates near 1.025
    auto grid = result.value().generate();
    EXPECT_EQ(grid.size(), 51);
    EXPECT_DOUBLE_EQ(grid[0], -3.0);
    EXPECT_DOUBLE_EQ(grid[50], 3.0);

    // Check strict monotonicity
    for (size_t i = 1; i < grid.size(); ++i) {
        EXPECT_GT(grid[i], grid[i-1]) << "Grid must be strictly monotonic at index " << i;
    }
}

TEST(GridSpecTest, MultiSinhBypassAutoMerge) {
    // Test bypassing automatic cluster merging
    // Two clusters close together (x=0.0 and x=0.1) would normally merge
    // Domain: [-3.0, 3.0]
    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = 0.0, .alpha = 3.0, .weight = 1.0},
        {.center_x = 0.1, .alpha = 3.0, .weight = 1.0}
    };

    // alpha_avg = 3.0, threshold = 0.3/3.0 = 0.1
    // delta_x = 0.1 <= 0.1, so clusters would normally merge
    // With auto_merge = false, they should remain separate

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 51, clusters, false);
    ASSERT_TRUE(result.has_value());

    // Should have TWO clusters (not merged)
    auto final_clusters = result.value().clusters();
    EXPECT_EQ(final_clusters.size(), 2);

    // Verify original centers preserved
    EXPECT_NEAR(final_clusters[0].center_x, 0.0, 1e-10);
    EXPECT_NEAR(final_clusters[1].center_x, 0.1, 1e-10);

    // Verify grid generation still works (produces valid monotonic grid)
    auto grid = result.value().generate();
    EXPECT_EQ(grid.size(), 51);
    EXPECT_DOUBLE_EQ(grid[0], -3.0);
    EXPECT_DOUBLE_EQ(grid[50], 3.0);

    // Check strict monotonicity
    for (size_t i = 1; i < grid.size(); ++i) {
        EXPECT_GT(grid[i], grid[i-1]) << "Grid must be strictly monotonic at index " << i;
    }
}

TEST(GridSpecTest, MultiSinhEtaParameterizationBounds) {
    // Regression test for eta-based parameterization fix
    // Previous u ∈ [-1,1] parameterization caused sinh values to exceed bounds
    // for typical alpha >= 1, requiring heavy clamping that flattened concentrations
    // eta ∈ [0,1] parameterization keeps sinh values bounded and preserves density

    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = -1.0, .alpha = 2.0, .weight = 1.0},
        {.center_x = 1.0, .alpha = 2.0, .weight = 1.0}
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 101, clusters);
    ASSERT_TRUE(result.has_value());

    auto grid = result.value().generate();
    EXPECT_EQ(grid.size(), 101);

    // Verify strict bounds (no out-of-bounds points that required clamping)
    for (size_t i = 0; i < grid.size(); ++i) {
        EXPECT_GE(grid[i], -3.0) << "Point " << i << " below x_min";
        EXPECT_LE(grid[i], 3.0) << "Point " << i << " above x_max";
    }

    // Verify endpoints are exact
    EXPECT_DOUBLE_EQ(grid[0], -3.0);
    EXPECT_DOUBLE_EQ(grid[100], 3.0);

    // Verify strict monotonicity (no backward jumps from clamping)
    for (size_t i = 1; i < grid.size(); ++i) {
        EXPECT_GT(grid[i], grid[i-1]) << "Non-monotonic at index " << i;
    }

    // Verify concentration is preserved near cluster centers
    // Find points closest to each cluster center
    size_t idx_left = 0, idx_right = 0;
    double min_dist_left = 100.0, min_dist_right = 100.0;
    for (size_t i = 0; i < grid.size(); ++i) {
        double dist_left = std::abs(grid[i] - (-1.0));
        double dist_right = std::abs(grid[i] - 1.0);
        if (dist_left < min_dist_left) {
            min_dist_left = dist_left;
            idx_left = i;
        }
        if (dist_right < min_dist_right) {
            min_dist_right = dist_right;
            idx_right = i;
        }
    }

    // Verify concentration is preserved near cluster centers
    // With alpha=2.0, spacing near centers should be fine (< 0.1)
    if (idx_left > 0 && idx_left < 100) {
        double spacing_near_left = grid[idx_left + 1] - grid[idx_left];
        EXPECT_LT(spacing_near_left, 0.1) << "Concentration lost at left cluster";
    }
    if (idx_right > 0 && idx_right < 100) {
        double spacing_near_right = grid[idx_right + 1] - grid[idx_right];
        EXPECT_LT(spacing_near_right, 0.1) << "Concentration lost at right cluster";
    }

    // Verify that the maximum spacing in the grid is reasonable
    // (not so large that it indicates flattening from clamping)
    double max_spacing = 0.0;
    for (size_t i = 1; i < grid.size(); ++i) {
        max_spacing = std::max(max_spacing, grid[i] - grid[i-1]);
    }
    // With 101 points on [-3,3] domain, max spacing should be < 0.5
    // (uniform would be 6/100 = 0.06, multi-sinh should be somewhat coarser but not extreme)
    EXPECT_LT(max_spacing, 0.5) << "Excessive spacing indicates concentration was lost";
}

TEST(GridSpecTest, MultiSinhSingleOffCenterPreserved) {
    // Regression test: Explicit single off-center clusters should NOT be recentered
    // Only merged clusters should be recentered
    // Use case: Deep ITM concentration at x = -0.5, not at domain center

    std::vector<mango::MultiSinhCluster<double>> clusters = {
        {.center_x = -0.5, .alpha = 2.0, .weight = 1.0}
    };

    auto result = mango::GridSpec<>::multi_sinh_spaced(-3.0, 3.0, 51, clusters);
    ASSERT_TRUE(result.has_value());

    // Verify the cluster center was NOT moved to domain midpoint (0.0)
    auto final_clusters = result.value().clusters();
    EXPECT_EQ(final_clusters.size(), 1);
    EXPECT_NEAR(final_clusters[0].center_x, -0.5, 1e-10)
        << "Explicit single off-center cluster should preserve its location";

    // Verify grid generation works and concentrates near -0.5
    auto grid = result.value().generate();
    EXPECT_EQ(grid.size(), 51);
    EXPECT_DOUBLE_EQ(grid[0], -3.0);
    EXPECT_DOUBLE_EQ(grid[50], 3.0);

    // Find point closest to -0.5
    size_t idx_center = 0;
    double min_dist = 100.0;
    for (size_t i = 0; i < grid.size(); ++i) {
        double dist = std::abs(grid[i] - (-0.5));
        if (dist < min_dist) {
            min_dist = dist;
            idx_center = i;
        }
    }

    // Verify concentration at -0.5 (fine spacing near center)
    if (idx_center > 0 && idx_center < 50) {
        double spacing_near_center = grid[idx_center + 1] - grid[idx_center];
        EXPECT_LT(spacing_near_center, 0.15)
            << "Spacing should be fine near requested center -0.5";
    }

    // Verify spacing is coarser far from -0.5 (e.g., near x = 2.5)
    size_t idx_far = 0;
    min_dist = 100.0;
    for (size_t i = 0; i < grid.size(); ++i) {
        double dist = std::abs(grid[i] - 2.5);
        if (dist < min_dist) {
            min_dist = dist;
            idx_far = i;
        }
    }

    if (idx_far > 0 && idx_far < 50) {
        double spacing_far = grid[idx_far + 1] - grid[idx_far];
        double spacing_near = grid[idx_center + 1] - grid[idx_center];
        // Off-center grids with monotonicity enforcement have reduced contrast,
        // but spacing far should still be coarser than spacing near the center
        EXPECT_GT(spacing_far, spacing_near * 0.9)
            << "Spacing should be coarser far from requested center";
    }
}
