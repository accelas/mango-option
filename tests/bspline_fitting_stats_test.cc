// tests/bspline_fitting_stats_test.cc
#include <gtest/gtest.h>
#include "src/option/table/price_table_builder.hpp"

TEST(BSplineFittingStatsTest, DefaultConstruction) {
    mango::BSplineFittingStats stats;
    EXPECT_EQ(stats.max_residual_axis0, 0.0);
    EXPECT_EQ(stats.max_residual_axis1, 0.0);
    EXPECT_EQ(stats.max_residual_axis2, 0.0);
    EXPECT_EQ(stats.max_residual_axis3, 0.0);
    EXPECT_EQ(stats.max_residual_overall, 0.0);
    EXPECT_EQ(stats.condition_axis0, 0.0);
    EXPECT_EQ(stats.condition_axis1, 0.0);
    EXPECT_EQ(stats.condition_axis2, 0.0);
    EXPECT_EQ(stats.condition_axis3, 0.0);
    EXPECT_EQ(stats.condition_max, 0.0);
    EXPECT_EQ(stats.failed_slices_axis0, 0);
    EXPECT_EQ(stats.failed_slices_axis1, 0);
    EXPECT_EQ(stats.failed_slices_axis2, 0);
    EXPECT_EQ(stats.failed_slices_axis3, 0);
    EXPECT_EQ(stats.failed_slices_total, 0);
}
