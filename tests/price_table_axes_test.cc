// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/bspline/bspline_surface.hpp"
#include <cmath>

namespace mango {
namespace {

TEST(PriceTableAxesTest, Create4DAxes) {
    PriceTableAxes axes;
    axes.grids[0] = {std::log(0.7), std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1), std::log(1.2), std::log(1.3)};
    axes.grids[1] = {0.027, 0.1, 0.5, 1.0, 2.0};
    axes.grids[2] = {0.10, 0.20, 0.30};
    axes.grids[3] = {0.0, 0.05, 0.10};

    axes.names[0] = "log_moneyness";
    axes.names[1] = "maturity";
    axes.names[2] = "volatility";
    axes.names[3] = "rate";

    EXPECT_EQ(axes.grids[0].size(), 7);
    EXPECT_EQ(axes.names[0], "log_moneyness");
}

TEST(PriceTableAxesTest, TotalGridPoints) {
    PriceTableAxes axes;
    axes.grids[0] = {std::log(0.7), std::log(0.8), std::log(0.9), std::log(1.0)};  // 4 points
    axes.grids[1] = {0.1, 0.5};            // 2 points
    axes.grids[2] = {0.10, 0.20, 0.30};    // 3 points
    axes.grids[3] = {0.0, 0.05};           // 2 points

    size_t total = axes.total_points();
    EXPECT_EQ(total, 4 * 2 * 3 * 2);  // 48 points
}

TEST(PriceTableAxesTest, ValidateMonotonic) {
    PriceTableAxesND<2> axes;
    axes.grids[0] = {1.0, 2.0, 3.0};
    axes.grids[1] = {0.1, 0.2, 0.3};

    auto result = axes.validate();
    EXPECT_TRUE(result.has_value());
}

TEST(PriceTableAxesTest, RejectNonMonotonic) {
    PriceTableAxesND<2> axes;
    axes.grids[0] = {1.0, 3.0, 2.0};  // Non-monotonic
    axes.grids[1] = {0.1, 0.2, 0.3};

    auto result = axes.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ValidationErrorCode::UnsortedGrid);
}

TEST(PriceTableAxesTest, RejectEmptyGrid) {
    PriceTableAxesND<2> axes;
    axes.grids[0] = {1.0, 2.0, 3.0};
    axes.grids[1] = {};  // Empty grid

    auto result = axes.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ValidationErrorCode::InvalidGridSize);
}

} // namespace
} // namespace mango
