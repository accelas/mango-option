#include "src/cpp/snapshot_interpolator.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

TEST(SnapshotInterpolatorTest, InterpolateParabola) {
    std::vector<double> x = {0.0, 0.25, 0.5, 0.75, 1.0};
    std::vector<double> y = {0.0, 0.0625, 0.25, 0.5625, 1.0};

    mango::SnapshotInterpolator interp;
    interp.build(std::span{x}, std::span{y});

    // Test interpolation (cubic splines have small error for parabolas)
    EXPECT_NEAR(interp.eval(0.125), 0.125*0.125, 1e-2);
    EXPECT_NEAR(interp.eval(0.5), 0.25, 1e-10);  // Exact at grid points
}

TEST(SnapshotInterpolatorTest, InterpolateFromData) {
    std::vector<double> x = {0.0, 0.5, 1.0};
    std::vector<double> y = {1.0, 2.0, 3.0};
    std::vector<double> dy = {2.0, 2.0, 2.0};  // Pre-computed derivative

    mango::SnapshotInterpolator interp;

    // Build from y data
    interp.build(std::span{x}, std::span{y});

    // Interpolate derivative from pre-computed data
    double deriv = interp.eval_from_data(0.5, std::span{dy});
    EXPECT_NEAR(deriv, 2.0, 1e-10);
}
