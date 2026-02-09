// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "boundary_detector.hpp"

using namespace mango;

TEST(BoundaryDetectorTest, EmptyCacheReturnsFallback) {
    PDESliceCache cache;
    std::vector<double> tau = {0.1, 0.5, 1.0};
    std::vector<double> sigma = {0.20};
    std::vector<double> rate = {0.05};

    auto result = detect_exercise_boundary(
        cache, tau, sigma, rate, 0, 2, 100.0, OptionType::PUT);

    // No valid slices → falls back to x*=0.0
    EXPECT_DOUBLE_EQ(result.x_star, 0.0);
    EXPECT_EQ(result.n_valid, 0u);
    EXPECT_EQ(result.n_sampled, 0u);
    EXPECT_DOUBLE_EQ(result.delta, 0.10);  // delta_min
}

TEST(BoundaryDetectorTest, EmptyCacheUsesFallbackXStar) {
    PDESliceCache cache;
    std::vector<double> tau = {0.1};
    std::vector<double> sigma = {0.20};
    std::vector<double> rate = {0.05};

    auto result = detect_exercise_boundary(
        cache, tau, sigma, rate, 0, 0, 100.0, OptionType::PUT,
        {}, 0.0, -0.15);  // fallback_x_star = -0.15

    EXPECT_DOUBLE_EQ(result.x_star, -0.15);
}
