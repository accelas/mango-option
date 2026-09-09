// SPDX-License-Identifier: MIT
#include "mango/math/proof/bspline_box.hpp"
#include <array>
#include <gtest/gtest.h>
using namespace mango::detail::proof;
TEST(ProofBSplineBoxTest, CoupledCoordinatesIncludeRawClampBranchesAndTheirPartials) {
    const std::array<double, 8> kx{-1, -1, -1, -1, 1, 1, 1, 1};
    const std::array<double, 8> ku{.25, .25, .25, .25, .75, .75, .75, .75};
    const std::array<double, 8> kz{-.5, -.5, -.5, -.5, .5, .5, .5, .5};
    const std::array<std::span<const double>, 3> knots{kx, ku, kz};
    std::vector<double> coefficients(64);
    for (std::size_t i = 0; i < 64; ++i) {
        const double u = .25 + .5 * ((i / 4) % 4) / 3;
        const double z = -.5 + (i % 4) / 3.;
        coefficients[i] = u - z + .5;
    }
    // f=clamp(u,.25,.75)-z+.5 at z=0, crossing both u clamps.
    const std::array<Interval, 3> box{Interval(0), Interval::hull(0, 1), Interval(0)};
    const std::array<std::size_t, 2> axes{1, 2};
    auto result = enclose_cubic_bspline_box(knots, coefficients, box, axes, 3);
    ASSERT_EQ(result.reason, StopReason::None);
    EXPECT_EQ(result.cells, 3u);
    EXPECT_NEAR(result.value.lower_bound(), .75, 1e-14);
    EXPECT_NEAR(result.value.upper_bound(), 1.25, 1e-14);
    ASSERT_EQ(result.partials.size(), 2u);
    EXPECT_EQ(result.partials[0].lower_bound(), 0);
    EXPECT_NEAR(result.partials[0].upper_bound(), 1, 1e-14);
    EXPECT_NEAR(result.partials[1].lower_bound(), -1, 1e-14);
    EXPECT_NEAR(result.partials[1].upper_bound(), -1, 1e-14);
    auto limited = enclose_cubic_bspline_box(knots, coefficients, box, axes, 2);
    EXPECT_EQ(limited.reason, StopReason::NodeBudget);
    EXPECT_EQ(limited.cells, 2u);
}
