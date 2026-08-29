// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/adaptive_grid_types.hpp"

TEST(AdaptiveGridParamsTest, DefaultMaxIterIsEight) {
    EXPECT_EQ(mango::AdaptiveGridParams{}.max_iter, 8u);
}

TEST(BuildDiagnosticsTest, DefaultsAreEmpty) {
    mango::BuildDiagnostics d;
    EXPECT_FALSE(d.target_met);
    EXPECT_EQ(d.holdout_points, 0u);
}
