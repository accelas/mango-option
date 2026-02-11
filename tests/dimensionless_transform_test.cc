// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/transforms/dimensionless_3d.hpp"
#include "mango/option/table/surface_concepts.hpp"
#include <cmath>

namespace mango {
namespace {

static_assert(CoordinateTransform<DimensionlessTransform3D>);

TEST(DimensionlessTransform3DTest, ToCoords) {
    DimensionlessTransform3D xform;
    auto c = xform.to_coords(100.0, 100.0, 1.0, 0.20, 0.05);
    EXPECT_NEAR(c[0], 0.0, 1e-14);
    EXPECT_NEAR(c[1], 0.02, 1e-14);
    EXPECT_NEAR(c[2], std::log(2.5), 1e-14);
}

TEST(DimensionlessTransform3DTest, ToCoordsOTM) {
    DimensionlessTransform3D xform;
    auto c = xform.to_coords(80.0, 100.0, 0.5, 0.30, 0.03);
    EXPECT_NEAR(c[0], std::log(0.8), 1e-14);
    EXPECT_NEAR(c[1], 0.30 * 0.30 * 0.5 / 2.0, 1e-14);
    EXPECT_NEAR(c[2], std::log(2.0 * 0.03 / (0.30 * 0.30)), 1e-14);
}

TEST(DimensionlessTransform3DTest, VegaWeights) {
    DimensionlessTransform3D xform;
    double tau = 1.0, sigma = 0.20;
    auto w = xform.vega_weights(100.0, 100.0, tau, sigma, 0.05);
    EXPECT_EQ(w[0], 0.0);
    EXPECT_NEAR(w[1], sigma * tau, 1e-14);
    EXPECT_NEAR(w[2], -2.0 / sigma, 1e-14);
}

TEST(DimensionlessTransform3DTest, KDimIs3) {
    EXPECT_EQ(DimensionlessTransform3D::kDim, 3u);
}

}  // namespace
}  // namespace mango
