// tests/bspline_fitting_stats_test.cc
#include <gtest/gtest.h>
#include "src/math/bspline_nd_separable.hpp"

TEST(BSplineFittingStatsTest, DefaultConstruction) {
    mango::BSplineFittingStats<double, 4> stats;

    // All per-axis arrays should be zero-initialized
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(stats.max_residual_per_axis[i], 0.0);
        EXPECT_EQ(stats.condition_per_axis[i], 0.0);
        EXPECT_EQ(stats.failed_slices_per_axis[i], 0);
    }

    // Overall stats should be zero
    EXPECT_EQ(stats.max_residual_overall, 0.0);
    EXPECT_EQ(stats.condition_max, 0.0);
    EXPECT_EQ(stats.failed_slices_total, 0);
}

TEST(BSplineFittingStatsTest, DifferentDimensions) {
    // Test with N=2
    mango::BSplineFittingStats<double, 2> stats2;
    EXPECT_EQ(stats2.max_residual_per_axis.size(), 2);
    EXPECT_EQ(stats2.condition_per_axis.size(), 2);
    EXPECT_EQ(stats2.failed_slices_per_axis.size(), 2);

    // Test with N=5
    mango::BSplineFittingStats<double, 5> stats5;
    EXPECT_EQ(stats5.max_residual_per_axis.size(), 5);
    EXPECT_EQ(stats5.condition_per_axis.size(), 5);
    EXPECT_EQ(stats5.failed_slices_per_axis.size(), 5);
}

TEST(BSplineFittingStatsTest, DifferentFloatTypes) {
    // Test with float
    mango::BSplineFittingStats<float, 3> stats_float;
    EXPECT_EQ(stats_float.max_residual_overall, 0.0f);

    // Test with long double
    mango::BSplineFittingStats<long double, 3> stats_ld;
    EXPECT_EQ(stats_ld.max_residual_overall, 0.0L);
}

TEST(BSplineNDSeparableResultTest, ToStatsComputesAggregates) {
    mango::BSplineNDSeparableResult<double, 3> result;
    result.coefficients = {1.0, 2.0, 3.0};
    result.max_residual_per_axis = {0.1, 0.3, 0.2};
    result.condition_per_axis = {10.0, 50.0, 30.0};
    result.failed_slices = {0, 2, 1};

    auto stats = result.to_stats();

    // Check per-axis values copied correctly
    EXPECT_EQ(stats.max_residual_per_axis[0], 0.1);
    EXPECT_EQ(stats.max_residual_per_axis[1], 0.3);
    EXPECT_EQ(stats.max_residual_per_axis[2], 0.2);

    // Check aggregates computed correctly
    EXPECT_EQ(stats.max_residual_overall, 0.3);  // max of {0.1, 0.3, 0.2}
    EXPECT_EQ(stats.condition_max, 50.0);        // max of {10, 50, 30}
    EXPECT_EQ(stats.failed_slices_total, 3);     // sum of {0, 2, 1}
}
